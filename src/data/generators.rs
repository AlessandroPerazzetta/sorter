/// Generate a vector of random integers
pub fn random_vec(size: usize) -> Vec<i32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..10000)).collect()
}

/// Generate a sorted vector of integers
pub fn sorted_vec(size: usize) -> Vec<i32> {
    (0..size as i32).collect()
}

/// Generate a reverse sorted vector of integers
pub fn reverse_sorted_vec(size: usize) -> Vec<i32> {
    (0..size as i32).rev().collect()
}

/// Generate a nearly sorted vector (small random swaps)
pub fn nearly_sorted_vec(size: usize) -> Vec<i32> {
    let mut vec = sorted_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

/// Generate a vector with many duplicates
pub fn duplicates_vec(size: usize) -> Vec<i32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..100)).collect()
}

/// Generate a vector with all identical elements
pub fn identical_vec(size: usize) -> Vec<i32> {
    vec![42; size]
}

/// Generate a vector with alternating high/low values
pub fn alternating_vec(size: usize) -> Vec<i32> {
    (0..size)
        .map(|i| if i % 2 == 0 { 0 } else { 10000 })
        .collect()
}

// Import SortableData and DataType for new generators
use crate::data::types::{DataType, SortableData};

/// Generate a vector of SortableData based on data type and pattern
pub fn generate_sortable_data(
    data_type: DataType,
    pattern: &str,
    size: usize,
) -> Vec<SortableData> {
    match data_type {
        DataType::I8 => generate_i8_data(pattern, size),
        DataType::I16 => generate_i16_data(pattern, size),
        DataType::I32 => generate_i32_data(pattern, size),
        DataType::I64 => generate_i64_data(pattern, size),
        DataType::U8 => generate_u8_data(pattern, size),
        DataType::U16 => generate_u16_data(pattern, size),
        DataType::U32 => generate_u32_data(pattern, size),
        DataType::U64 => generate_u64_data(pattern, size),
        DataType::F32 => generate_f32_data(pattern, size),
        DataType::F64 => generate_f64_data(pattern, size),
        DataType::String => generate_string_data(pattern, size),
    }
}

fn generate_i8_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_i8_vec(size),
        "sorted" => sorted_i8_vec(size),
        "reverse" => reverse_sorted_i8_vec(size),
        "nearly" => nearly_sorted_i8_vec(size),
        "duplicates" => duplicates_i8_vec(size),
        "identical" => identical_i8_vec(size),
        "alternating" => alternating_i8_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter().map(SortableData::I8).collect()
}

fn generate_i16_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_i16_vec(size),
        "sorted" => sorted_i16_vec(size),
        "reverse" => reverse_sorted_i16_vec(size),
        "nearly" => nearly_sorted_i16_vec(size),
        "duplicates" => duplicates_i16_vec(size),
        "identical" => identical_i16_vec(size),
        "alternating" => alternating_i16_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter().map(SortableData::I16).collect()
}

fn generate_i32_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_vec(size),
        "sorted" => sorted_vec(size),
        "reverse" => reverse_sorted_vec(size),
        "nearly" => nearly_sorted_vec(size),
        "duplicates" => duplicates_vec(size),
        "identical" => identical_vec(size),
        "alternating" => alternating_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter().map(SortableData::I32).collect()
}

fn generate_i64_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_i64_vec(size),
        "sorted" => sorted_i64_vec(size),
        "reverse" => reverse_sorted_i64_vec(size),
        "nearly" => nearly_sorted_i64_vec(size),
        "duplicates" => duplicates_i64_vec(size),
        "identical" => identical_i64_vec(size),
        "alternating" => alternating_i64_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter().map(SortableData::I64).collect()
}

fn generate_u8_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_u8_vec(size),
        "sorted" => sorted_u8_vec(size),
        "reverse" => reverse_sorted_u8_vec(size),
        "nearly" => nearly_sorted_u8_vec(size),
        "duplicates" => duplicates_u8_vec(size),
        "identical" => identical_u8_vec(size),
        "alternating" => alternating_u8_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter().map(SortableData::U8).collect()
}

fn generate_u16_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_u16_vec(size),
        "sorted" => sorted_u16_vec(size),
        "reverse" => reverse_sorted_u16_vec(size),
        "nearly" => nearly_sorted_u16_vec(size),
        "duplicates" => duplicates_u16_vec(size),
        "identical" => identical_u16_vec(size),
        "alternating" => alternating_u16_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter().map(SortableData::U16).collect()
}

fn generate_u32_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_u32_vec(size),
        "sorted" => sorted_u32_vec(size),
        "reverse" => reverse_sorted_u32_vec(size),
        "nearly" => nearly_sorted_u32_vec(size),
        "duplicates" => duplicates_u32_vec(size),
        "identical" => identical_u32_vec(size),
        "alternating" => alternating_u32_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter().map(SortableData::U32).collect()
}

fn generate_u64_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_u64_vec(size),
        "sorted" => sorted_u64_vec(size),
        "reverse" => reverse_sorted_u64_vec(size),
        "nearly" => nearly_sorted_u64_vec(size),
        "duplicates" => duplicates_u64_vec(size),
        "identical" => identical_u64_vec(size),
        "alternating" => alternating_u64_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter().map(SortableData::U64).collect()
}

fn generate_f32_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_f32_vec(size),
        "sorted" => sorted_f32_vec(size),
        "reverse" => reverse_sorted_f32_vec(size),
        "nearly" => nearly_sorted_f32_vec(size),
        "duplicates" => duplicates_f32_vec(size),
        "identical" => identical_f32_vec(size),
        "alternating" => alternating_f32_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter()
        .map(|v| SortableData::F32(crate::OrderedF32(v)))
        .collect()
}

fn generate_f64_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_f64_vec(size),
        "sorted" => sorted_f64_vec(size),
        "reverse" => reverse_sorted_f64_vec(size),
        "nearly" => nearly_sorted_f64_vec(size),
        "duplicates" => duplicates_f64_vec(size),
        "identical" => identical_f64_vec(size),
        "alternating" => alternating_f64_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter()
        .map(|v| SortableData::F64(crate::OrderedF64(v)))
        .collect()
}

fn generate_string_data(pattern: &str, size: usize) -> Vec<SortableData> {
    let data = match pattern {
        "random" => random_string_vec(size),
        "sorted" => sorted_string_vec(size),
        "reverse" => reverse_sorted_string_vec(size),
        "nearly" => nearly_sorted_string_vec(size),
        "duplicates" => duplicates_string_vec(size),
        "identical" => identical_string_vec(size),
        "alternating" => alternating_string_vec(size),
        _ => panic!("Unknown pattern: {}", pattern),
    };
    data.into_iter().map(SortableData::String).collect()
}

// Helper functions for generating different data types
fn random_i8_vec(size: usize) -> Vec<i8> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-50..50)).collect()
}

fn sorted_i8_vec(size: usize) -> Vec<i8> {
    (0..size as i8).collect()
}

fn reverse_sorted_i8_vec(size: usize) -> Vec<i8> {
    (0..size as i8).rev().collect()
}

fn nearly_sorted_i8_vec(size: usize) -> Vec<i8> {
    let mut vec = sorted_i8_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

fn duplicates_i8_vec(size: usize) -> Vec<i8> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-10..10)).collect()
}

fn identical_i8_vec(size: usize) -> Vec<i8> {
    vec![42; size]
}

fn alternating_i8_vec(size: usize) -> Vec<i8> {
    (0..size)
        .map(|i| if i % 2 == 0 { -50 } else { 50 })
        .collect()
}

fn random_i16_vec(size: usize) -> Vec<i16> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-1000..1000)).collect()
}

fn sorted_i16_vec(size: usize) -> Vec<i16> {
    (0..size as i16).collect()
}

fn reverse_sorted_i16_vec(size: usize) -> Vec<i16> {
    (0..size as i16).rev().collect()
}

fn nearly_sorted_i16_vec(size: usize) -> Vec<i16> {
    let mut vec = sorted_i16_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

fn duplicates_i16_vec(size: usize) -> Vec<i16> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-100..100)).collect()
}

fn identical_i16_vec(size: usize) -> Vec<i16> {
    vec![42; size]
}

fn alternating_i16_vec(size: usize) -> Vec<i16> {
    (0..size)
        .map(|i| if i % 2 == 0 { -1000 } else { 1000 })
        .collect()
}

fn random_i64_vec(size: usize) -> Vec<i64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-100000..100000)).collect()
}

fn sorted_i64_vec(size: usize) -> Vec<i64> {
    (0..size as i64).collect()
}

fn reverse_sorted_i64_vec(size: usize) -> Vec<i64> {
    (0..size as i64).rev().collect()
}

fn nearly_sorted_i64_vec(size: usize) -> Vec<i64> {
    let mut vec = sorted_i64_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

fn duplicates_i64_vec(size: usize) -> Vec<i64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-1000..1000)).collect()
}

fn identical_i64_vec(size: usize) -> Vec<i64> {
    vec![42; size]
}

fn alternating_i64_vec(size: usize) -> Vec<i64> {
    (0..size)
        .map(|i| if i % 2 == 0 { -100000 } else { 100000 })
        .collect()
}

fn random_u8_vec(size: usize) -> Vec<u8> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..100)).collect()
}

fn sorted_u8_vec(size: usize) -> Vec<u8> {
    (0..size as u8).collect()
}

fn reverse_sorted_u8_vec(size: usize) -> Vec<u8> {
    (0..size as u8).rev().collect()
}

fn nearly_sorted_u8_vec(size: usize) -> Vec<u8> {
    let mut vec = sorted_u8_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

fn duplicates_u8_vec(size: usize) -> Vec<u8> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..10)).collect()
}

fn identical_u8_vec(size: usize) -> Vec<u8> {
    vec![42; size]
}

fn alternating_u8_vec(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| if i % 2 == 0 { 0 } else { 100 })
        .collect()
}

fn random_u16_vec(size: usize) -> Vec<u16> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..1000)).collect()
}

fn sorted_u16_vec(size: usize) -> Vec<u16> {
    (0..size as u16).collect()
}

fn reverse_sorted_u16_vec(size: usize) -> Vec<u16> {
    (0..size as u16).rev().collect()
}

fn nearly_sorted_u16_vec(size: usize) -> Vec<u16> {
    let mut vec = sorted_u16_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

fn duplicates_u16_vec(size: usize) -> Vec<u16> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..100)).collect()
}

fn identical_u16_vec(size: usize) -> Vec<u16> {
    vec![42; size]
}

fn alternating_u16_vec(size: usize) -> Vec<u16> {
    (0..size)
        .map(|i| if i % 2 == 0 { 0 } else { 1000 })
        .collect()
}

fn random_u32_vec(size: usize) -> Vec<u32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..10000)).collect()
}

fn sorted_u32_vec(size: usize) -> Vec<u32> {
    (0..size as u32).collect()
}

fn reverse_sorted_u32_vec(size: usize) -> Vec<u32> {
    (0..size as u32).rev().collect()
}

fn nearly_sorted_u32_vec(size: usize) -> Vec<u32> {
    let mut vec = sorted_u32_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

fn duplicates_u32_vec(size: usize) -> Vec<u32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..100)).collect()
}

fn identical_u32_vec(size: usize) -> Vec<u32> {
    vec![42; size]
}

fn alternating_u32_vec(size: usize) -> Vec<u32> {
    (0..size)
        .map(|i| if i % 2 == 0 { 0 } else { 10000 })
        .collect()
}

fn random_u64_vec(size: usize) -> Vec<u64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..100000)).collect()
}

fn sorted_u64_vec(size: usize) -> Vec<u64> {
    (0..size as u64).collect()
}

fn reverse_sorted_u64_vec(size: usize) -> Vec<u64> {
    (0..size as u64).rev().collect()
}

fn nearly_sorted_u64_vec(size: usize) -> Vec<u64> {
    let mut vec = sorted_u64_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

fn duplicates_u64_vec(size: usize) -> Vec<u64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..1000)).collect()
}

fn identical_u64_vec(size: usize) -> Vec<u64> {
    vec![42; size]
}

fn alternating_u64_vec(size: usize) -> Vec<u64> {
    (0..size)
        .map(|i| if i % 2 == 0 { 0 } else { 100000 })
        .collect()
}

fn random_f32_vec(size: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-100.0..100.0)).collect()
}

fn sorted_f32_vec(size: usize) -> Vec<f32> {
    (0..size).map(|i| i as f32).collect()
}

fn reverse_sorted_f32_vec(size: usize) -> Vec<f32> {
    (0..size).map(|i| (size - i - 1) as f32).collect()
}

fn nearly_sorted_f32_vec(size: usize) -> Vec<f32> {
    let mut vec = sorted_f32_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

fn duplicates_f32_vec(size: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect()
}

fn identical_f32_vec(size: usize) -> Vec<f32> {
    vec![42.0; size]
}

fn alternating_f32_vec(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| if i % 2 == 0 { -100.0 } else { 100.0 })
        .collect()
}

fn random_f64_vec(size: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-100.0..100.0)).collect()
}

fn sorted_f64_vec(size: usize) -> Vec<f64> {
    (0..size).map(|i| i as f64).collect()
}

fn reverse_sorted_f64_vec(size: usize) -> Vec<f64> {
    (0..size).map(|i| (size - i - 1) as f64).collect()
}

fn nearly_sorted_f64_vec(size: usize) -> Vec<f64> {
    let mut vec = sorted_f64_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

fn duplicates_f64_vec(size: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-10.0..10.0)).collect()
}

fn identical_f64_vec(size: usize) -> Vec<f64> {
    vec![42.0; size]
}

fn alternating_f64_vec(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| if i % 2 == 0 { -100.0 } else { 100.0 })
        .collect()
}

fn random_string_vec(size: usize) -> Vec<String> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| format!("item{:04}", rng.gen_range(0..1000)))
        .collect()
}

fn sorted_string_vec(size: usize) -> Vec<String> {
    (0..size).map(|i| format!("item{:04}", i)).collect()
}

fn reverse_sorted_string_vec(size: usize) -> Vec<String> {
    (0..size)
        .map(|i| format!("item{:04}", size - i - 1))
        .collect()
}

fn nearly_sorted_string_vec(size: usize) -> Vec<String> {
    let mut vec = sorted_string_vec(size);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(size / 10).max(1) {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        vec.swap(i, j);
    }
    vec
}

fn duplicates_string_vec(size: usize) -> Vec<String> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| format!("item{:02}", rng.gen_range(0..10)))
        .collect()
}

fn identical_string_vec(size: usize) -> Vec<String> {
    vec!["item0042".to_string(); size]
}

fn alternating_string_vec(size: usize) -> Vec<String> {
    (0..size)
        .map(|i| {
            if i % 2 == 0 {
                "item0000".to_string()
            } else {
                "item9999".to_string()
            }
        })
        .collect()
}
