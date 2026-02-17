/// Enum to represent different data types that can be sorted
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SortableData {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(OrderedF32),
    F64(OrderedF64),
    String(String),
}

/// PartialOrd implementation: Since SortableData implements Ord (total ordering),
/// PartialOrd delegates to Ord for consistency and to avoid clippy warnings.
/// This is the canonical implementation for types that have total ordering.
impl PartialOrd for SortableData {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Ord implementation: Provides total ordering for the enum by comparing discriminants first,
/// then the inner values. This ensures consistent ordering across different data types.
impl Ord for SortableData {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_order = self.discriminant_order();
        let other_order = other.discriminant_order();
        self_order
            .cmp(&other_order)
            .then_with(|| match (self, other) {
                (SortableData::I8(a), SortableData::I8(b)) => a.cmp(b),
                (SortableData::I16(a), SortableData::I16(b)) => a.cmp(b),
                (SortableData::I32(a), SortableData::I32(b)) => a.cmp(b),
                (SortableData::I64(a), SortableData::I64(b)) => a.cmp(b),
                (SortableData::U8(a), SortableData::U8(b)) => a.cmp(b),
                (SortableData::U16(a), SortableData::U16(b)) => a.cmp(b),
                (SortableData::U32(a), SortableData::U32(b)) => a.cmp(b),
                (SortableData::U64(a), SortableData::U64(b)) => a.cmp(b),
                (SortableData::F32(a), SortableData::F32(b)) => a.cmp(b),
                (SortableData::F64(a), SortableData::F64(b)) => a.cmp(b),
                (SortableData::String(a), SortableData::String(b)) => a.cmp(b),
                _ => unreachable!(),
            })
    }
}

impl SortableData {
    /// Returns a discriminant order for comparison purposes.
    fn discriminant_order(&self) -> usize {
        match self {
            SortableData::I8(_) => 0,
            SortableData::I16(_) => 1,
            SortableData::I32(_) => 2,
            SortableData::I64(_) => 3,
            SortableData::U8(_) => 4,
            SortableData::U16(_) => 5,
            SortableData::U32(_) => 6,
            SortableData::U64(_) => 7,
            SortableData::F32(_) => 8,
            SortableData::F64(_) => 9,
            SortableData::String(_) => 10,
        }
    }
}

/// Wrapper for f32 to implement Ord
#[derive(Debug, Clone, Copy)]
pub struct OrderedF32(pub f32);

impl PartialEq for OrderedF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedF32 {}

/// PartialOrd implementation: Since OrderedF32 implements Ord (total ordering),
/// PartialOrd delegates to Ord for consistency and to avoid clippy warnings.
/// This is the canonical implementation for types that have total ordering.
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Ord implementation: Provides total ordering for f32 by handling NaN values.
/// NaN is treated as equal to other NaN for sorting purposes, ensuring no incomparable values.
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.0.partial_cmp(&other.0) {
            Some(ord) => ord,
            None => {
                // Handle NaN case: treat all NaN as equal for sorting purposes
                // This ensures total ordering
                std::cmp::Ordering::Equal
            }
        }
    }
}

/// Wrapper for f64 to implement Ord
#[derive(Debug, Clone, Copy)]
pub struct OrderedF64(pub f64);

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedF64 {}

/// PartialOrd implementation: Since OrderedF64 implements Ord (total ordering),
/// PartialOrd delegates to Ord for consistency and to avoid clippy warnings.
/// This is the canonical implementation for types that have total ordering.
impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Ord implementation: Provides total ordering for f64 by handling NaN values.
/// NaN is treated as equal to other NaN for sorting purposes, ensuring no incomparable values.
impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.0.partial_cmp(&other.0) {
            Some(ord) => ord,
            None => {
                // Handle NaN case: treat all NaN as equal for sorting purposes
                // This ensures total ordering
                std::cmp::Ordering::Equal
            }
        }
    }
}

impl std::fmt::Display for SortableData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SortableData::I8(v) => write!(f, "{}", v),
            SortableData::I16(v) => write!(f, "{}", v),
            SortableData::I32(v) => write!(f, "{}", v),
            SortableData::I64(v) => write!(f, "{}", v),
            SortableData::U8(v) => write!(f, "{}", v),
            SortableData::U16(v) => write!(f, "{}", v),
            SortableData::U32(v) => write!(f, "{}", v),
            SortableData::U64(v) => write!(f, "{}", v),
            SortableData::F32(v) => write!(f, "{}", v.0),
            SortableData::F64(v) => write!(f, "{}", v.0),
            SortableData::String(v) => write!(f, "{}", v),
        }
    }
}

/// Data type specification for parsing and generation
#[derive(Debug, Clone, Copy)]
pub enum DataType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    String,
}

impl DataType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "i8" => Some(DataType::I8),
            "i16" => Some(DataType::I16),
            "i32" => Some(DataType::I32),
            "i64" => Some(DataType::I64),
            "u8" => Some(DataType::U8),
            "u16" => Some(DataType::U16),
            "u32" => Some(DataType::U32),
            "u64" => Some(DataType::U64),
            "f32" => Some(DataType::F32),
            "f64" => Some(DataType::F64),
            "string" | "str" => Some(DataType::String),
            _ => None,
        }
    }

    pub fn parse_from_args(arg: &str) -> Result<Self, String> {
        Self::from_str(arg).ok_or_else(|| {
            format!(
                "Invalid data type '{}'. Supported types: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, string",
                arg
            )
        })
    }

    /// Returns the string representation of this data type
    pub fn as_str(&self) -> &'static str {
        match self {
            DataType::I8 => "i8",
            DataType::I16 => "i16",
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::U8 => "u8",
            DataType::U16 => "u16",
            DataType::U32 => "u32",
            DataType::U64 => "u64",
            DataType::F32 => "f32",
            DataType::F64 => "f64",
            DataType::String => "string",
        }
    }
}
