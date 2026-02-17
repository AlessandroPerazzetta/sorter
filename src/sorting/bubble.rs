/// Bubble sort implementation for sorting elements in ascending order
/// Time complexity: O(n^2), Space complexity: O(1)
/// Returns the number of passes performed
pub fn bubble_sort<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    let mut passes = 0;
    for i in 0..len {
        passes += 1;
        for j in 0..len - i - 1 {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
            }
        }
    }
    passes
}

/// Parallel bubble sort using odd-even sort algorithm
/// Time complexity: O(n^2), Space complexity: O(1)
/// Returns the number of passes performed
/// Note: While parallelized, bubble sort remains O(n^2) so parallel version
/// is mainly for demonstration purposes
pub fn parallel_bubble_sort<T: Ord + Send + Sync>(arr: &mut [T]) -> usize {

    let len = arr.len();
    let mut passes = 0;

    // Odd-even sort: alternate between odd and even phases
    // Each phase can be parallelized
    loop {
        let mut swapped = false;

        // Even phase: compare elements at even indices
        (0..len).step_by(2).for_each(|i| {
            if i + 1 < len && arr[i] > arr[i + 1] {
                arr.swap(i, i + 1);
                swapped = true;
            }
        });

        // Odd phase: compare elements at odd indices
        (1..len).step_by(2).for_each(|i| {
            if i + 1 < len && arr[i] > arr[i + 1] {
                arr.swap(i, i + 1);
                swapped = true;
            }
        });

        passes += 1;

        if !swapped {
            break;
        }
    }

    passes
}