use rayon;

/// Quicksort implementation for sorting elements in ascending order
/// Time complexity: O(n log n) average, O(n^2) worst case, Space complexity: O(log n)
/// Returns 0 (no pass count for this algorithm)
pub fn quicksort<T: Ord>(arr: &mut [T]) -> usize {
    if arr.len() <= 1 {
        return 0;
    }
    let pivot_index = partition(arr);
    quicksort(&mut arr[0..pivot_index]);
    quicksort(&mut arr[pivot_index + 1..]);
    0
}

/// Parallel quicksort implementation using Rayon
/// Time complexity: O(n log n) average, O(n^2) worst case, Space complexity: O(log n)
/// Returns 0 (no pass count for this algorithm)
pub fn parallel_quicksort<T: Ord + Send>(arr: &mut [T]) -> usize {
    parallel_quicksort_recursive(arr, 2000); // Threshold for parallelism
    0
}

fn parallel_quicksort_recursive<T: Ord + Send>(arr: &mut [T], threshold: usize) {
    if arr.len() <= 1 {
        return;
    }
    
    if arr.len() <= threshold {
        // Use sequential sort for small arrays
        quicksort(arr);
        return;
    }
    
    let pivot_index = partition(arr);
    let (left, right) = arr.split_at_mut(pivot_index);
    let right = &mut right[1..]; // Skip the pivot element
    
    // Check for unbalanced partitioning (which happens with identical elements)
    // If one side is much larger than the other, fall back to sequential
    let left_len = left.len();
    let right_len = right.len();
    let total_len = left_len + right_len;
    
    if left_len > total_len * 3 / 4 || right_len > total_len * 3 / 4 {
        // Unbalanced partition, use sequential sort
        quicksort(arr);
        return;
    }
    
    // Parallel recursive calls
    rayon::join(
        || parallel_quicksort_recursive(left, threshold),
        || parallel_quicksort_recursive(right, threshold)
    );
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    
    // Median of three pivot selection
    let mid = len / 2;
    let last = len - 1;
    
    // Sort first, middle, and last elements
    if arr[0] > arr[mid] { arr.swap(0, mid); }
    if arr[0] > arr[last] { arr.swap(0, last); }
    if arr[mid] > arr[last] { arr.swap(mid, last); }
    
    // Pivot is now at mid
    arr.swap(mid, len - 1);
    let mut i = 0;
    for j in 0..len - 1 {
        if arr[j] <= arr[len - 1] {
            arr.swap(i, j);
            i += 1;
        }
    }
    arr.swap(i, len - 1);
    i
}