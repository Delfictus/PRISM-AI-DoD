//! KD-Tree for Efficient K-Nearest Neighbor Search
//!
//! Implements a space-partitioning data structure for fast k-NN queries
//! in the KSG (Kraskov-Stögbauer-Grassberger) transfer entropy estimator.
//!
//! Complexity:
//! - Construction: O(N log N)
//! - k-NN Query: O(log N) average, O(N) worst case
//! - Memory: O(N)

use std::cmp::Ordering;

/// KD-Tree node
#[derive(Debug, Clone)]
struct KdNode {
    /// Point coordinates
    point: Vec<f64>,
    /// Original index in dataset
    index: usize,
    /// Split dimension
    split_dim: usize,
    /// Left subtree
    left: Option<Box<KdNode>>,
    /// Right subtree
    right: Option<Box<KdNode>>,
}

/// KD-Tree for efficient nearest neighbor search
#[derive(Debug, Clone)]
pub struct KdTree {
    /// Root node
    root: Option<Box<KdNode>>,
    /// Dimensionality
    dimensions: usize,
    /// Number of points
    n_points: usize,
}

/// Neighbor result from k-NN query
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbor {
    /// Index of neighbor
    pub index: usize,
    /// Distance to neighbor
    pub distance: f64,
}

impl KdTree {
    /// Construct KD-tree from points
    ///
    /// # Arguments
    /// * `points` - Matrix of points [n_points x dimensions]
    ///
    /// # Returns
    /// KD-tree with O(N log N) construction time
    pub fn new(points: &[Vec<f64>]) -> Self {
        if points.is_empty() {
            return Self {
                root: None,
                dimensions: 0,
                n_points: 0,
            };
        }

        let dimensions = points[0].len();
        let n_points = points.len();

        // Create indices for points
        let mut indices: Vec<(Vec<f64>, usize)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i))
            .collect();

        let root = Self::build_tree(&mut indices, 0, dimensions);

        Self {
            root,
            dimensions,
            n_points,
        }
    }

    /// Recursively build KD-tree
    fn build_tree(points: &mut [(Vec<f64>, usize)], depth: usize, dimensions: usize) -> Option<Box<KdNode>> {
        if points.is_empty() {
            return None;
        }

        if points.len() == 1 {
            let (point, index) = points[0].clone();
            return Some(Box::new(KdNode {
                point,
                index,
                split_dim: depth % dimensions,
                left: None,
                right: None,
            }));
        }

        // Choose split dimension (cycle through dimensions)
        let split_dim = depth % dimensions;

        // Sort by split dimension
        points.sort_by(|a, b| {
            a.0[split_dim].partial_cmp(&b.0[split_dim]).unwrap_or(Ordering::Equal)
        });

        // Find median
        let median_idx = points.len() / 2;

        let (median_point, median_index) = points[median_idx].clone();

        // Recursively build subtrees
        let left = Self::build_tree(&mut points[..median_idx], depth + 1, dimensions);
        let right = Self::build_tree(&mut points[median_idx + 1..], depth + 1, dimensions);

        Some(Box::new(KdNode {
            point: median_point,
            index: median_index,
            split_dim,
            left,
            right,
        }))
    }

    /// Find k nearest neighbors using L-infinity (max) norm
    ///
    /// This is the norm used in KSG estimator
    ///
    /// # Arguments
    /// * `query` - Query point
    /// * `k` - Number of neighbors
    ///
    /// # Returns
    /// Vector of k nearest neighbors sorted by distance
    pub fn knn_search(&self, query: &[f64], k: usize) -> Vec<Neighbor> {
        if self.root.is_none() || k == 0 {
            return Vec::new();
        }

        let mut neighbors = Vec::new();
        let mut max_dist = f64::INFINITY;

        self.knn_search_recursive(
            self.root.as_ref().unwrap(),
            query,
            k,
            &mut neighbors,
            &mut max_dist,
        );

        // Sort by distance
        neighbors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));

        neighbors
    }

    /// Recursive k-NN search
    fn knn_search_recursive(
        &self,
        node: &KdNode,
        query: &[f64],
        k: usize,
        neighbors: &mut Vec<Neighbor>,
        max_dist: &mut f64,
    ) {
        // Calculate distance to current node (L-infinity norm)
        let dist = self.distance_linf(&node.point, query);

        // Update neighbors if this point is closer
        if neighbors.len() < k {
            neighbors.push(Neighbor {
                index: node.index,
                distance: dist,
            });

            if neighbors.len() == k {
                // Update max_dist when we first fill the neighbor list
                *max_dist = neighbors.iter().map(|n| n.distance).fold(f64::NEG_INFINITY, f64::max);
            }
        } else if dist < *max_dist {
            // Replace farthest neighbor
            let max_idx = neighbors
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.distance.partial_cmp(&b.1.distance).unwrap_or(Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap();

            neighbors[max_idx] = Neighbor {
                index: node.index,
                distance: dist,
            };

            // Update max_dist
            *max_dist = neighbors.iter().map(|n| n.distance).fold(f64::NEG_INFINITY, f64::max);
        }

        // Determine which subtree to search first
        let split_dim = node.split_dim;
        let diff = query[split_dim] - node.point[split_dim];

        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Search first subtree
        if let Some(child) = first {
            self.knn_search_recursive(child, query, k, neighbors, max_dist);
        }

        // Check if we need to search second subtree
        // For L-infinity norm, we need to check if the splitting plane
        // could contain closer points
        if diff.abs() < *max_dist || neighbors.len() < k {
            if let Some(child) = second {
                self.knn_search_recursive(child, query, k, neighbors, max_dist);
            }
        }
    }

    /// L-infinity (max) norm distance
    ///
    /// d(x, y) = max_i |x_i - y_i|
    ///
    /// This is the metric used in KSG transfer entropy estimator
    fn distance_linf(&self, p1: &[f64], p2: &[f64]) -> f64 {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Range search: find all points within distance epsilon
    ///
    /// Used for counting neighbors in KSG estimator
    pub fn range_search(&self, query: &[f64], epsilon: f64) -> Vec<usize> {
        if self.root.is_none() {
            return Vec::new();
        }

        let mut neighbors = Vec::new();
        self.range_search_recursive(self.root.as_ref().unwrap(), query, epsilon, &mut neighbors);
        neighbors
    }

    /// Recursive range search
    fn range_search_recursive(
        &self,
        node: &KdNode,
        query: &[f64],
        epsilon: f64,
        neighbors: &mut Vec<usize>,
    ) {
        // Check if current node is within range
        let dist = self.distance_linf(&node.point, query);
        if dist <= epsilon {
            neighbors.push(node.index);
        }

        // Determine if we need to search subtrees
        let split_dim = node.split_dim;
        let diff = query[split_dim] - node.point[split_dim];

        // Search left subtree
        if let Some(child) = &node.left {
            if diff - epsilon <= 0.0 {
                self.range_search_recursive(child, query, epsilon, neighbors);
            }
        }

        // Search right subtree
        if let Some(child) = &node.right {
            if diff + epsilon >= 0.0 {
                self.range_search_recursive(child, query, epsilon, neighbors);
            }
        }
    }

    /// Get number of points in tree
    pub fn len(&self) -> usize {
        self.n_points
    }

    /// Check if tree is empty
    pub fn is_empty(&self) -> bool {
        self.n_points == 0
    }

    /// Get dimensionality
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_construction() {
        let points = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        let tree = KdTree::new(&points);
        assert_eq!(tree.len(), 4);
        assert_eq!(tree.dimensions(), 2);
    }

    #[test]
    fn test_knn_search() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
        ];

        let tree = KdTree::new(&points);

        // Query point near (0, 0)
        let query = vec![0.1, 0.1];
        let neighbors = tree.knn_search(&query, 2);

        assert_eq!(neighbors.len(), 2);
        // Should find (0,0) and either (1,0) or (0,1) as nearest
        assert!(neighbors[0].distance < 0.2);
    }

    #[test]
    fn test_linf_distance() {
        let tree = KdTree::new(&[vec![0.0, 0.0]]);

        let p1 = vec![0.0, 0.0];
        let p2 = vec![1.0, 0.5];

        // L-inf distance should be max(|1-0|, |0.5-0|) = 1.0
        let dist = tree.distance_linf(&p1, &p2);
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_range_search() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
        ];

        let tree = KdTree::new(&points);

        // Search within distance 1.1 of origin
        let query = vec![0.0, 0.0];
        let neighbors = tree.range_search(&query, 1.1);

        // Should find (0,0), (1,0), (0,1), and (1,1)
        assert!(neighbors.len() >= 3);
        assert!(neighbors.contains(&0)); // (0,0)
    }

    #[test]
    fn test_empty_tree() {
        let tree = KdTree::new(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);

        let neighbors = tree.knn_search(&vec![0.0], 1);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_single_point() {
        let points = vec![vec![1.0, 2.0, 3.0]];
        let tree = KdTree::new(&points);

        let neighbors = tree.knn_search(&vec![1.5, 2.5, 3.5], 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].index, 0);
    }

    #[test]
    fn test_high_dimensional() {
        // Test with higher dimensionality (typical for embedded time series)
        let points: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let angle = i as f64 * 0.1;
                vec![angle.cos(), angle.sin(), angle * 0.5, (angle * 2.0).sin()]
            })
            .collect();

        let tree = KdTree::new(&points);
        assert_eq!(tree.dimensions(), 4);

        let query = vec![0.5, 0.5, 0.3, 0.2];
        let neighbors = tree.knn_search(&query, 5);
        assert_eq!(neighbors.len(), 5);

        // Verify neighbors are sorted by distance
        for i in 0..neighbors.len() - 1 {
            assert!(neighbors[i].distance <= neighbors[i + 1].distance);
        }
    }
}
