/// Selection sort implementation for sorting elements in ascending order
/// Time complexity: O(n^2), Space complexity: O(1)
/// Returns 0 (no pass count for this algorithm)
pub fn selection_sort<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    for i in 0..len {
        let mut min_idx = i;
        for j in (i + 1)..len {
            if arr[j] < arr[min_idx] {
                min_idx = j;
            }
        }
        arr.swap(i, min_idx);
    }
    0
}

/// Parallel selection sort
/// Time complexity: O(n^2), Space complexity: O(1)
/// Returns 0 (no pass count for this algorithm)
/// Note: While parallelized, selection sort remains O(n^2) so parallel version
/// is mainly for demonstration purposes
pub fn parallel_selection_sort<T: Ord + Send + Sync>(arr: &mut [T]) -> usize {
    use rayon::prelude::*;

    let len = arr.len();
    for i in 0..len {
        // Find the minimum element in the unsorted portion
        // We can parallelize this search, though it's still O(n) per iteration
        let min_idx = (i..len)
            .into_par_iter()
            .min_by_key(|&j| &arr[j])
            .unwrap();

        arr.swap(i, min_idx);
    }
    0
}