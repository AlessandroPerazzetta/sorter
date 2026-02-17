/// Insertion sort implementation for sorting elements in ascending order
/// Time complexity: O(n^2), Space complexity: O(1)
/// Returns 0 (no pass count for this algorithm)
pub fn insertion_sort<T: Ord>(arr: &mut [T]) -> usize {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j] < arr[j - 1] {
            arr.swap(j, j - 1);
            j -= 1;
        }
    }
    0
}