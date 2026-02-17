use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use std::time::Instant;

mod data;
mod sorting;

// Re-export types for convenience
use data::types::{DataType, OrderedF32, OrderedF64, SortableData};

fn load_data_from_file(
    filename: &str,
    data_type: DataType,
) -> Result<Vec<SortableData>, Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);

    let mut data = Vec::new();
    let mut line_number = 0;

    for line in reader.lines() {
        line_number += 1;
        let line = line?;
        let line = line.trim();

        // Skip empty lines and comments (lines starting with #)
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse the value based on data type
        let sortable_data = match data_type {
            DataType::I8 => match line.parse::<i8>() {
                Ok(num) => SortableData::I8(num),
                Err(_) => {
                    return Err(format!(
                        "Invalid i8 '{}' on line {} in file '{}'",
                        line, line_number, filename
                    )
                    .into());
                }
            },
            DataType::I16 => match line.parse::<i16>() {
                Ok(num) => SortableData::I16(num),
                Err(_) => {
                    return Err(format!(
                        "Invalid i16 '{}' on line {} in file '{}'",
                        line, line_number, filename
                    )
                    .into());
                }
            },
            DataType::I32 => match line.parse::<i32>() {
                Ok(num) => SortableData::I32(num),
                Err(_) => {
                    return Err(format!(
                        "Invalid i32 '{}' on line {} in file '{}'",
                        line, line_number, filename
                    )
                    .into());
                }
            },
            DataType::I64 => match line.parse::<i64>() {
                Ok(num) => SortableData::I64(num),
                Err(_) => {
                    return Err(format!(
                        "Invalid i64 '{}' on line {} in file '{}'",
                        line, line_number, filename
                    )
                    .into());
                }
            },
            DataType::U8 => match line.parse::<u8>() {
                Ok(num) => SortableData::U8(num),
                Err(_) => {
                    return Err(format!(
                        "Invalid u8 '{}' on line {} in file '{}'",
                        line, line_number, filename
                    )
                    .into());
                }
            },
            DataType::U16 => match line.parse::<u16>() {
                Ok(num) => SortableData::U16(num),
                Err(_) => {
                    return Err(format!(
                        "Invalid u16 '{}' on line {} in file '{}'",
                        line, line_number, filename
                    )
                    .into());
                }
            },
            DataType::U32 => match line.parse::<u32>() {
                Ok(num) => SortableData::U32(num),
                Err(_) => {
                    return Err(format!(
                        "Invalid u32 '{}' on line {} in file '{}'",
                        line, line_number, filename
                    )
                    .into());
                }
            },
            DataType::U64 => match line.parse::<u64>() {
                Ok(num) => SortableData::U64(num),
                Err(_) => {
                    return Err(format!(
                        "Invalid u64 '{}' on line {} in file '{}'",
                        line, line_number, filename
                    )
                    .into());
                }
            },
            DataType::F32 => match line.parse::<f32>() {
                Ok(num) => SortableData::F32(OrderedF32(num)),
                Err(_) => {
                    return Err(format!(
                        "Invalid f32 '{}' on line {} in file '{}'",
                        line, line_number, filename
                    )
                    .into());
                }
            },
            DataType::F64 => match line.parse::<f64>() {
                Ok(num) => SortableData::F64(OrderedF64(num)),
                Err(_) => {
                    return Err(format!(
                        "Invalid f64 '{}' on line {} in file '{}'",
                        line, line_number, filename
                    )
                    .into());
                }
            },
            DataType::String => SortableData::String(line.to_string()),
        };

        data.push(sortable_data);
    }

    if data.is_empty() {
        return Err(format!("No valid data found in file '{}'", filename).into());
    }

    println!("Loaded {} values from file: {}", data.len(), filename);
    Ok(data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.len() < 3 || args.len() > 10 {
        println!(
            "Usage: sorter [--parallel] [--data-type <type>] [--save-data <file>] [--load-data <file>] [--save-results <dir>] <algorithm> <data_pattern> <size>"
        );
        println!(
            "       sorter [--data-type <type>] [--save-data <file>] [--load-data <file>] [--save-results <dir>] <algorithm> <data_pattern> <size>"
        );
        println!("Algorithms: bubble, selection, insertion, merge, quick, heap");
        println!(
            "           parallel-bubble, parallel-selection, parallel-merge, parallel-quick, parallel-heap"
        );
        println!("           gpu-sort, gpu-bitonic (GPU-accelerated)");
        println!(
            "Data patterns: random, sorted, reverse, nearly, duplicates, identical, alternating"
        );
        println!(
            "Data types: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, string (default: i32)"
        );
        println!("Options: --parallel (use parallel algorithms where available)");
        println!("         --data-type <type> (specify data type for generation/loading)");
        println!("         --save-data <file> (save generated data to file)");
        println!("         --load-data <file> (load data from file instead of generating)");
        println!("         --save-results <dir> (save sorting results as JSON to directory)");
        println!("Example: sorter bubble random 1000");
        println!("Example: sorter --data-type f64 bubble random 1000");
        println!("Example: sorter --parallel --data-type string merge random 10000");
        println!("Example: sorter --save-data data.txt bubble random 1000");
        println!("Example: sorter --load-data data.txt --data-type i32 bubble random 1000");
        println!("Example: sorter --save-results results/ bubble random 1000");
        return Ok(());
    }

    // Parse arguments
    let mut parallel = false;
    let mut data_type = DataType::I32; // Default data type
    let mut save_data_file: Option<String> = None;
    let mut load_data_file: Option<String> = None;
    let mut save_results_dir: Option<String> = None;
    let mut positional_args = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--parallel" => {
                parallel = true;
                i += 1;
            }
            "--data-type" => {
                if i + 1 < args.len() {
                    data_type = match DataType::parse_from_args(&args[i + 1]) {
                        Ok(dt) => dt,
                        Err(err) => {
                            println!("Error: {}", err);
                            return Ok(());
                        }
                    };
                    i += 2;
                } else {
                    println!("Error: --data-type requires a type argument");
                    return Ok(());
                }
            }
            "--save-data" => {
                if i + 1 < args.len() {
                    save_data_file = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    println!("Error: --save-data requires a filename argument");
                    return Ok(());
                }
            }
            "--load-data" => {
                if i + 1 < args.len() {
                    load_data_file = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    println!("Error: --load-data requires a filename argument");
                    return Ok(());
                }
            }
            "--save-results" => {
                if i + 1 < args.len() {
                    save_results_dir = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    println!("Error: --save-results requires a directory argument");
                    return Ok(());
                }
            }
            _ => {
                positional_args.push(args[i].clone());
                i += 1;
            }
        }
    }

    // Check we have the right number of positional arguments
    if positional_args.len() != 3 {
        println!(
            "Invalid arguments. Expected 3 positional arguments: algorithm, data_pattern, size"
        );
        println!("Got: {:?}", positional_args);
        return Ok(());
    }

    let algorithm = &positional_args[0];
    let data_pattern = &positional_args[1];
    let size_str = &positional_args[2];

    // Check if algorithm is a parallel variant (overrides --parallel flag)
    let is_parallel_alg = matches!(
        algorithm.as_str(),
        "parallel-bubble"
            | "parallel-selection"
            | "parallel-merge"
            | "parallel-quick"
            | "parallel-heap"
    );
    if is_parallel_alg {
        parallel = true;
    }

    let size: usize = size_str
        .parse()
        .map_err(|_| "Invalid size: must be a positive integer")?;

    // Load or generate data
    let mut data = if let Some(filename) = &load_data_file {
        // Load data from file
        load_data_from_file(filename, data_type)?
    } else {
        // Generate data based on data_type and data_pattern
        data::generators::generate_sortable_data(data_type, data_pattern, size)
    };

    // Verify data size matches expected size (only when not loading from file)
    if load_data_file.is_none() && data.len() != size {
        println!(
            "Warning: Generated data size ({}) doesn't match requested size ({})",
            data.len(),
            size
        );
    } else if load_data_file.is_some() && data.len() != size {
        println!(
            "Warning: Loaded data size ({}) doesn't match requested size ({}). Using actual loaded size.",
            data.len(),
            size
        );
    }

    // Save data to file if requested
    if let Some(filename) = &save_data_file {
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create(filename)?;
        for item in &data {
            writeln!(file, "{}", item)?;
        }
        println!("Generated data saved to: {}", filename);
    }

    // Sort and measure
    let start = Instant::now();
    let passes = match (parallel, algorithm.as_str()) {
        (false, "bubble") => sorting::bubble::bubble_sort(&mut data),
        (false, "selection") => sorting::selection::selection_sort(&mut data),
        (false, "insertion") => sorting::insertion::insertion_sort(&mut data),
        (false, "merge") => sorting::merge::mergesort(&mut data),
        (false, "quick") => sorting::quick::quicksort(&mut data),
        (false, "heap") => sorting::heap::heap_sort(&mut data),
        (true, "bubble") | (true, "parallel-bubble") => {
            sorting::bubble::parallel_bubble_sort(&mut data)
        }
        (true, "selection") | (true, "parallel-selection") => {
            sorting::selection::parallel_selection_sort(&mut data)
        }
        (true, "merge") | (true, "parallel-merge") => sorting::merge::parallel_mergesort(&mut data),
        (true, "quick") | (true, "parallel-quick") => sorting::quick::parallel_quicksort(&mut data),
        (true, "heap") | (true, "parallel-heap") => sorting::heap::parallel_heap_sort(&mut data),
        (false, "gpu-sort") => sorting::gpu::gpu_sort(&mut data),
        (false, "gpu-bitonic") => sorting::gpu::gpu_bitonic_sort(&mut data),
        _ => {
            println!("Unknown algorithm or parallel mode not supported for this algorithm.");
            println!("Available sequential: bubble, selection, insertion, merge, quick, heap");
            println!(
                "Available parallel: bubble (parallel-bubble), selection (parallel-selection), merge (parallel-merge), quick (parallel-quick), heap (parallel-heap)"
            );
            println!("Available GPU: gpu-sort, gpu-bitonic");
            return Ok(());
        }
    };
    let duration = start.elapsed();

    // Output results
    let algo_type = if algorithm.starts_with("gpu-") {
        "GPU-accelerated"
    } else if parallel || algorithm.starts_with("parallel-") {
        "parallel"
    } else {
        "sequential"
    };
    println!("Algorithm: {} ({})", algorithm, algo_type);

    if let Some(filename) = &load_data_file {
        println!("Data source: Loaded from file ({})", filename);
        println!("Data size: {}", data.len());
    } else {
        println!("Data type: {:?}", data_type);
        println!("Data pattern: {}", data_pattern);
        println!("Size: {}", size);
    }

    println!("Time taken: {:.6} seconds", duration.as_secs_f64());
    if passes > 0 {
        println!("Passes: {}", passes);
    }
    println!("First 10 elements: {:?}", &data[..data.len().min(10)]);
    println!(
        "Last 10 elements: {:?}",
        &data[data.len().saturating_sub(10)..]
    );

    // Save results to JSON file if requested
    if let Some(dir_path) = &save_results_dir {
        use std::fs;
        use std::path::Path;

        // Create directory if it doesn't exist
        if let Err(e) = fs::create_dir_all(dir_path) {
            println!("Error creating results directory '{}': {}", dir_path, e);
            return Ok(());
        }

        // Generate filename based on algorithm, data pattern, size, and timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let filename = format!(
            "{}_{}_{}_{}_{}.json",
            algorithm,
            data_pattern,
            size,
            data_type.as_str(),
            timestamp
        );
        let filepath = Path::new(dir_path).join(filename);

        // Create JSON structure
        let result = serde_json::json!({
            "algorithm": algorithm,
            "data_type": format!("{:?}", data_type),
            "data_pattern": data_pattern,
            "size": size,
            "execution_time_seconds": duration.as_secs_f64(),
            "passes": passes,
            "parallel": parallel || is_parallel_alg,
            "gpu_accelerated": algorithm.starts_with("gpu-"),
            "timestamp": timestamp,
            "sorted_data": data.iter().map(|item| format!("{}", item)).collect::<Vec<_>>()
        });

        // Write to file
        match fs::write(&filepath, result.to_string()) {
            Ok(_) => println!("Results saved to: {}", filepath.display()),
            Err(e) => println!("Error saving results to '{}': {}", filepath.display(), e),
        }
    }

    Ok(())
}
