/// Heap sort implementation for sorting elements in ascending order
/// Time complexity: O(n log n), Space complexity: O(1)
/// Returns 0 (no pass count for this algorithm)
pub fn heap_sort<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    // Build max heap
    for i in (0..len / 2).rev() {
        heapify(arr, len, i);
    }
    // Extract elements from heap
    for i in (1..len).rev() {
        arr.swap(0, i);
        heapify(arr, i, 0);
    }
    0
}

/// Parallel heap sort implementation using Rayon
/// Time complexity: O(n log n), Space complexity: O(1)
/// Returns 0 (no pass count for this algorithm)
/// Note: Heap sort parallelization is complex due to dependencies.
/// This implementation uses sequential heap sort for now.
pub fn parallel_heap_sort<T: Ord + Send + Sync>(arr: &mut [T]) -> usize {
    // For now, use the sequential implementation
    // True parallel heap sort would require more complex coordination
    heap_sort(arr)
}

fn heapify<T: Ord>(arr: &mut [T], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;
    if left < n && arr[left] > arr[largest] {
        largest = left;
    }
    if right < n && arr[right] > arr[largest] {
        largest = right;
    }
    if largest != i {
        arr.swap(i, largest);
        heapify(arr, n, largest);
    }
}