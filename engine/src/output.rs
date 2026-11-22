//! Writing the two output tables as Parquet files (one row group each).

use parquet::data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub enum Column {
    /// Strings; with definition levels (1 present, 0 null) for a nullable column,
    /// in which case only the present values are listed.
    Str(Vec<ByteArray>, Option<Vec<i16>>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
}

/// Write the columns, in schema order, to path.
pub fn write(path: &Path, schema: &str, columns: Vec<Column>) -> parquet::errors::Result<()> {
    let schema = Arc::new(parse_message_type(schema)?);
    // No dictionary encoding: nearly every user_id is distinct, so building a
    // dictionary is wasted work (and the small columns don't need one).
    let props = Arc::new(WriterProperties::builder().set_dictionary_enabled(false).build());
    let mut writer = SerializedFileWriter::new(File::create(path)?, schema, props)?;
    let mut group = writer.next_row_group()?;
    for column in columns {
        let mut w = group.next_column()?.expect("schema has a column for each one written");
        match column {
            Column::Str(v, def) => {
                w.typed::<ByteArrayType>().write_batch(&v, def.as_deref(), None)?;
            }
            Column::Int32(v) => {
                w.typed::<Int32Type>().write_batch(&v, None, None)?;
            }
            Column::Int64(v) => {
                w.typed::<Int64Type>().write_batch(&v, None, None)?;
            }
        }
        w.close()?;
    }
    group.close()?;
    writer.close()?;
    Ok(())
}

pub const USER_DAILY: &str = "
message user_daily {
  REQUIRED BYTE_ARRAY user_id (UTF8);
  REQUIRED INT32 day (DATE);
  REQUIRED INT64 sessions;
  REQUIRED INT64 events;
  REQUIRED INT64 revenue_cents;
  OPTIONAL BYTE_ARRAY top_category (UTF8);
}";

pub const CAMPAIGN_DAILY: &str = "
message campaign_daily {
  REQUIRED BYTE_ARRAY campaign_id (UTF8);
  REQUIRED BYTE_ARRAY channel (UTF8);
  REQUIRED INT32 day (DATE);
  REQUIRED INT64 sessions;
  REQUIRED INT64 purchasing_sessions;
  REQUIRED INT64 revenue_cents;
  REQUIRED INT64 assisted_sessions;
}";
