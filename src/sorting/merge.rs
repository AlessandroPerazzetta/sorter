use rayon;

/// Mergesort implementation for sorting elements in ascending order
/// Time complexity: O(n log n), Space complexity: O(n)
/// Returns 0 (no pass count for this algorithm)
pub fn mergesort<T: Ord + Clone>(arr: &mut [T]) -> usize {
    let len = arr.len();
    if len <= 1 {
        return 0;
    }
    let mid = len / 2;
    mergesort(&mut arr[0..mid]);
    mergesort(&mut arr[mid..]);
    let result = merge(&arr[0..mid], &arr[mid..]);
    for (i, item) in result.into_iter().enumerate() {
        arr[i] = item;
    }
    0
}

/// Parallel mergesort implementation using Rayon
/// Time complexity: O(n log n), Space complexity: O(n)
/// Returns 0 (no pass count for this algorithm)
pub fn parallel_mergesort<T: Ord + Clone + Send>(arr: &mut [T]) -> usize {
    parallel_mergesort_recursive(arr, 1000); // Threshold for parallelism
    0
}

fn parallel_mergesort_recursive<T: Ord + Clone + Send>(arr: &mut [T], threshold: usize) {
    let len = arr.len();
    if len <= 1 {
        return;
    }

    if len <= threshold {
        // Use sequential sort for small arrays
        mergesort(arr);
        return;
    }

    let mid = len / 2;
    let (left, right) = arr.split_at_mut(mid);

    // Parallel recursive calls
    rayon::join(
        || parallel_mergesort_recursive(left, threshold),
        || parallel_mergesort_recursive(right, threshold),
    );

    // Merge the sorted halves
    let result = merge(left, right);
    for (i, item) in result.into_iter().enumerate() {
        arr[i] = item;
    }
}

fn merge<T: Ord + Clone>(left: &[T], right: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            result.push(left[i].clone());
            i += 1;
        } else {
            result.push(right[j].clone());
            j += 1;
        }
    }
    result.extend_from_slice(&left[i..]);
    result.extend_from_slice(&right[j..]);
    result
}
