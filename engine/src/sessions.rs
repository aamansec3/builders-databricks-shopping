//! Sessions, user_daily rows, click credit and assists: one partition of people
//! per task. Each task sorts its people's events and clicks by (person, ts)
//! and walks each person's events once.
//!
//! A session ends where the gap to the next event is over 30 minutes; equal
//! timestamps have gap 0, so they always share a session. As sessions advance,
//! two pointers keep the person's clicks with start - 7d <= ts <= start in a
//! window, with a count per campaign: the campaigns with a count are the
//! assists, and the window's latest click (smallest click_id on a ts tie) gets
//! the credit; an empty window credits organic (campaign index ncamp).

use crate::parse::{DAY, bytes};
use rayon::prelude::*;

pub const GAP: i64 = 1_800_000;
pub const WEEK: i64 = 7 * DAY;

/// One user_daily row; top is a category index, or -1 with no purchases.
pub struct UserDay {
    pub person: u32,
    pub day: i32,
    pub sessions: i64,
    pub events: i64,
    pub revenue: i64,
    pub top: i16,
}

/// campaign_daily sums, indexed [campaign * ndays + day - day0].
pub struct Acc {
    pub ndays: usize,
    pub sessions: Vec<i64>,
    pub purchasing: Vec<i64>,
    pub revenue: Vec<i64>,
    pub assisted: Vec<i64>,
}

impl Acc {
    fn new(ncamp: usize, ndays: usize) -> Acc {
        let n = (ncamp + 1) * ndays;
        Acc { ndays, sessions: vec![0; n], purchasing: vec![0; n], revenue: vec![0; n], assisted: vec![0; n] }
    }
    fn add(&mut self, o: &Acc) {
        for (a, b) in [
            (&mut self.sessions, &o.sessions),
            (&mut self.purchasing, &o.purchasing),
            (&mut self.revenue, &o.revenue),
            (&mut self.assisted, &o.assisted),
        ] {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
    }
}

/// What the walk needs beyond the records themselves.
pub struct Input {
    pub eshift: u32,
    pub cshift: u32,
    pub ws: i64,
    pub ncat: usize,
    pub ncamp: usize,
    pub day0: i64,
    pub ndays: usize,
}

/// A kept event: (key, pay). key is person << eshift | (ts - window start), so
/// sorting keys sorts by (person, ts); pay is price << 16 | category << 1 | is_purchase.
pub type SessEvent = (i64, i64);
/// A valid click: (key, campaign, click_id length, click_id pointer), with key
/// person << cshift | (ts - (window start - 7 days)).
pub type SessClick = (i64, u32, u32, u64);

/// Run every person partition: events[q] and clicks[q] are partition q's
/// records, as one list per thread that produced them.
pub fn run(inp: &Input, events: Vec<Vec<Vec<SessEvent>>>, clicks: Vec<Vec<Vec<SessClick>>>) -> (Vec<UserDay>, Acc) {
    events
        .into_par_iter()
        .zip(clicks)
        .fold(
            || (Vec::new(), Acc::new(inp.ncamp, inp.ndays)),
            |(mut rows, mut acc), (ev, cv)| {
                partition(inp, &mut ev.concat(), &mut cv.concat(), &mut rows, &mut acc);
                (rows, acc)
            },
        )
        .reduce_with(|(mut r1, mut a1), (r2, a2)| {
            r1.extend(r2);
            a1.add(&a2);
            (r1, a1)
        })
        .unwrap_or_else(|| (Vec::new(), Acc::new(inp.ncamp, inp.ndays)))
}

fn partition(inp: &Input, ev: &mut [SessEvent], cv: &mut [SessClick], rows: &mut Vec<UserDay>, acc: &mut Acc) {
    if ev.is_empty() {
        return;
    }
    ev.sort_unstable_by_key(|e| e.0);
    cv.sort_unstable_by_key(|c| c.0);

    let (es, cs) = (inp.eshift, inp.cshift);
    let emask = (1i64 << es) - 1;
    let cmask = (1i64 << cs) - 1;
    let ets = |k: usize| (ev[k].0 & emask) + inp.ws;
    let cts = |k: usize| (cv[k].0 & cmask) + inp.ws - WEEK;
    let nd = acc.ndays;

    let mut cnt = vec![0u32; inp.ncamp]; // clicks in the window, per campaign
    let mut active: Vec<u32> = Vec::with_capacity(inp.ncamp); // campaigns with cnt > 0
    let mut slot = vec![0u32; inp.ncamp]; // where each is in active
    let mut catrev = vec![0i64; inp.ncat];
    let (m, mc) = (ev.len(), cv.len());
    let (mut i, mut ci) = (0, 0);
    while i < m {
        let p = ev[i].0 >> es;
        let mut j = i;
        while j < m && ev[j].0 >> es == p {
            j += 1;
        }
        while ci < mc && cv[ci].0 >> cs < p {
            ci += 1;
        }
        let mut ce = ci;
        while ce < mc && cv[ce].0 >> cs == p {
            ce += 1;
        }
        let (mut lo, mut hi) = (ci, ci);
        active.clear();
        let mut day = -1i64;
        let (mut d_sess, mut d_events, mut d_rev) = (0i64, 0i64, 0i64);
        let mut k = i;
        while k < j {
            let start = ets(k);
            let sday = start.div_euclid(DAY);
            if sday != day {
                if day >= 0 {
                    emit(rows, p, day, d_sess, d_events, d_rev, &mut catrev);
                }
                day = sday;
                (d_sess, d_events, d_rev) = (0, 0, 0);
            }
            let (mut n_ev, mut rev, mut purch) = (0i64, 0i64, false);
            let mut prev = start;
            while k < j && ets(k) - prev <= GAP {
                let pay = ev[k].1;
                n_ev += 1;
                if pay & 1 == 1 {
                    let price = pay >> 16;
                    rev += price;
                    purch = true;
                    catrev[((pay >> 1) & 0x7FFF) as usize] += price;
                }
                prev = ets(k);
                k += 1;
            }

            while hi < ce && cts(hi) <= start {
                let c = cv[hi].1 as usize;
                if cnt[c] == 0 {
                    slot[c] = active.len() as u32;
                    active.push(c as u32);
                }
                cnt[c] += 1;
                hi += 1;
            }
            while lo < hi && cts(lo) < start - WEEK {
                let c = cv[lo].1 as usize;
                cnt[c] -= 1;
                if cnt[c] == 0 {
                    let last = active.pop().unwrap();
                    if last as usize != c {
                        active[slot[c] as usize] = last;
                        slot[last as usize] = slot[c];
                    }
                }
                lo += 1;
            }

            let c = if hi > lo {
                let mut b = hi - 1;
                let tb = cts(b);
                let mut x = hi - 1;
                while x > lo {
                    x -= 1;
                    if cts(x) != tb {
                        break;
                    }
                    if bytes(cv[x].3, cv[x].2) < bytes(cv[b].3, cv[b].2) {
                        b = x;
                    }
                }
                cv[b].1 as usize
            } else {
                inp.ncamp
            };
            let d = (sday - inp.day0) as usize;
            acc.sessions[c * nd + d] += 1;
            acc.purchasing[c * nd + d] += purch as i64;
            acc.revenue[c * nd + d] += rev;
            for &a in &active {
                acc.assisted[a as usize * nd + d] += 1;
            }
            d_sess += 1;
            d_events += n_ev;
            d_rev += rev;
        }
        emit(rows, p, day, d_sess, d_events, d_rev, &mut catrev);
        for &a in &active {
            cnt[a as usize] = 0;
        }
        i = j;
    }
}

/// One user_daily row; clears the category sums.
fn emit(rows: &mut Vec<UserDay>, p: i64, day: i64, sessions: i64, events: i64, revenue: i64, catrev: &mut [i64]) {
    let mut top = -1i16;
    let mut best = 0i64;
    for (c, &v) in catrev.iter().enumerate() {
        if v > 0 && (top < 0 || v > best) {
            top = c as i16; // categories are numbered in byte order, so a tie keeps the smallest
            best = v;
        }
    }
    catrev.fill(0);
    rows.push(UserDay { person: p as u32, day: day as i32, sessions, events, revenue, top });
}
