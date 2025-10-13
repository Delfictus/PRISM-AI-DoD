//! KD-Tree for Efficient Nearest Neighbor Search
//!
//! Implements a balanced KD-tree for O(N log N) construction and O(log N) query
//! Used for Kraskov-Stögbauer-Grassberger (KSG) Transfer Entropy estimation
//!
//! # Mathematical Foundation
//!
//! KD-tree partitions k-dimensional space using axis-aligned hyperplanes
//! Enables efficient range queries and k-NN search for entropy estimation
//!
//! # Reference
//! Bentley, J. L. (1975). "Multidimensional binary search trees used for associative searching"

use ndarray::Array1;

/// Point in k-dimensional space
#[derive(Debug, Clone)]
pub struct Point {
    pub coords: Vec<f64>,
    pub index: usize,
}

impl Point {
    pub fn new(coords: Vec<f64>, index: usize) -> Self {
        Self { coords, index }
    }

    /// Euclidean distance to another point
    pub fn distance(&self, other: &Point) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Maximum (Chebyshev) distance to another point
    pub fn max_distance(&self, other: &Point) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }
}

/// KD-Tree node
#[derive(Debug, Clone)]
enum KdNode {
    Leaf {
        point: Point,
    },
    Internal {
        point: Point,
        left: Box<KdNode>,
        right: Box<KdNode>,
        split_dim: usize,
        split_value: f64,
    },
}

/// KD-Tree for efficient nearest neighbor search
#[derive(Debug, Clone)]
pub struct KdTree {
    root: Option<Box<KdNode>>,
    dimensions: usize,
    size: usize,
}

impl KdTree {
    /// Create a new KD-tree from points
    pub fn new(points: Vec<Point>) -> Self {
        if points.is_empty() {
            return Self {
                root: None,
                dimensions: 0,
                size: 0,
            };
        }

        let dimensions = points[0].coords.len();
        let size = points.len();
        let root = Some(Box::new(Self::build_recursive(points, 0, dimensions)));

        Self {
            root,
            dimensions,
            size,
        }
    }

    /// Recursively build KD-tree
    fn build_recursive(mut points: Vec<Point>, depth: usize, dimensions: usize) -> KdNode {
        if points.is_empty() {
            panic!("Cannot build tree from empty points");
        }

        if points.len() == 1 {
            return KdNode::Leaf {
                point: points.into_iter().next().unwrap(),
            };
        }

        // Choose split dimension (cycle through dimensions)
        let split_dim = depth % dimensions;

        // Sort points by split dimension
        points.sort_by(|a, b| {
            a.coords[split_dim]
                .partial_cmp(&b.coords[split_dim])
                .unwrap()
        });

        // Find median
        let median_idx = points.len() / 2;
        let median_point = points[median_idx].clone();
        let split_value = median_point.coords[split_dim];

        // Split into left and right
        let left_points: Vec<Point> = points.iter().take(median_idx).cloned().collect();
        let right_points: Vec<Point> = points.iter().skip(median_idx + 1).cloned().collect();

        // Recursively build subtrees
        let left = if !left_points.is_empty() {
            Box::new(Self::build_recursive(left_points, depth + 1, dimensions))
        } else {
            Box::new(KdNode::Leaf {
                point: median_point.clone(),
            })
        };

        let right = if !right_points.is_empty() {
            Box::new(Self::build_recursive(right_points, depth + 1, dimensions))
        } else {
            Box::new(KdNode::Leaf {
                point: median_point.clone(),
            })
        };

        KdNode::Internal {
            point: median_point,
            left,
            right,
            split_dim,
            split_value,
        }
    }

    /// Find k-nearest neighbors using Euclidean distance
    pub fn k_nearest(&self, query: &Point, k: usize) -> Vec<(f64, usize)> {
        if self.root.is_none() || k == 0 {
            return Vec::new();
        }

        let mut neighbors = Vec::new();
        self.k_nearest_recursive(&self.root.as_ref().unwrap(), query, k, &mut neighbors);

        // Sort by distance and return k nearest
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        neighbors.into_iter().take(k).collect()
    }

    /// Recursively search for k-nearest neighbors
    fn k_nearest_recursive(
        &self,
        node: &KdNode,
        query: &Point,
        k: usize,
        neighbors: &mut Vec<(f64, usize)>,
    ) {
        match node {
            KdNode::Leaf { point } => {
                let dist = query.distance(point);
                neighbors.push((dist, point.index));
            }
            KdNode::Internal {
                point,
                left,
                right,
                split_dim,
                split_value,
            } => {
                // Add current point
                let dist = query.distance(point);
                neighbors.push((dist, point.index));

                // Determine which subtree to search first
                let query_val = query.coords[*split_dim];
                let (first, second) = if query_val < *split_value {
                    (left, right)
                } else {
                    (right, left)
                };

                // Search closer subtree
                self.k_nearest_recursive(first, query, k, neighbors);

                // Check if we need to search the other subtree
                // Only search if the distance to the splitting plane is less than
                // the distance to the k-th nearest neighbor
                if neighbors.len() < k {
                    self.k_nearest_recursive(second, query, k, neighbors);
                } else {
                    // Sort to find k-th nearest distance
                    neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    let kth_dist = neighbors[k.min(neighbors.len()) - 1].0;
                    let plane_dist = (query_val - *split_value).abs();

                    if plane_dist < kth_dist {
                        self.k_nearest_recursive(second, query, k, neighbors);
                    }
                }
            }
        }
    }

    /// Range query: find all points within distance epsilon
    pub fn range_query(&self, query: &Point, epsilon: f64, use_max_norm: bool) -> Vec<usize> {
        if self.root.is_none() {
            return Vec::new();
        }

        let mut results = Vec::new();
        self.range_query_recursive(
            &self.root.as_ref().unwrap(),
            query,
            epsilon,
            &mut results,
            use_max_norm,
        );
        results
    }

    /// Recursively search for points in range
    fn range_query_recursive(
        &self,
        node: &KdNode,
        query: &Point,
        epsilon: f64,
        results: &mut Vec<usize>,
        use_max_norm: bool,
    ) {
        match node {
            KdNode::Leaf { point } => {
                let dist = if use_max_norm {
                    query.max_distance(point)
                } else {
                    query.distance(point)
                };

                if dist <= epsilon {
                    results.push(point.index);
                }
            }
            KdNode::Internal {
                point,
                left,
                right,
                split_dim,
                split_value,
            } => {
                // Check current point
                let dist = if use_max_norm {
                    query.max_distance(point)
                } else {
                    query.distance(point)
                };

                if dist <= epsilon {
                    results.push(point.index);
                }

                // Determine which subtrees to search
                let query_val = query.coords[*split_dim];
                let plane_dist = (query_val - *split_value).abs();

                // Always search closer subtree
                if query_val < *split_value {
                    self.range_query_recursive(left, query, epsilon, results, use_max_norm);
                    // Search other subtree if plane is within epsilon
                    if plane_dist <= epsilon {
                        self.range_query_recursive(right, query, epsilon, results, use_max_norm);
                    }
                } else {
                    self.range_query_recursive(right, query, epsilon, results, use_max_norm);
                    // Search other subtree if plane is within epsilon
                    if plane_dist <= epsilon {
                        self.range_query_recursive(left, query, epsilon, results, use_max_norm);
                    }
                }
            }
        }
    }

    /// Get tree size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get tree dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_creation() {
        let points = vec![
            Point::new(vec![1.0, 2.0], 0),
            Point::new(vec![3.0, 4.0], 1),
            Point::new(vec![5.0, 6.0], 2),
        ];

        let tree = KdTree::new(points);
        assert_eq!(tree.size(), 3);
        assert_eq!(tree.dimensions(), 2);
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let points = vec![
            Point::new(vec![0.0, 0.0], 0),
            Point::new(vec![1.0, 0.0], 1),
            Point::new(vec![0.0, 1.0], 2),
            Point::new(vec![5.0, 5.0], 3),
        ];

        let tree = KdTree::new(points);
        let query = Point::new(vec![0.5, 0.5], 999);
        let neighbors = tree.k_nearest(&query, 2);

        assert_eq!(neighbors.len(), 2);
        // Closest should be indices 1 and 2
        let indices: Vec<usize> = neighbors.iter().map(|(_, idx)| *idx).collect();
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
    }

    #[test]
    fn test_range_query() {
        let points = vec![
            Point::new(vec![0.0, 0.0], 0),
            Point::new(vec![1.0, 0.0], 1),
            Point::new(vec![0.0, 1.0], 2),
            Point::new(vec![5.0, 5.0], 3),
        ];

        let tree = KdTree::new(points);
        let query = Point::new(vec![0.0, 0.0], 999);
        let results = tree.range_query(&query, 1.5, false);

        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(!results.contains(&3));
    }

    #[test]
    fn test_max_distance() {
        let p1 = Point::new(vec![0.0, 0.0, 0.0], 0);
        let p2 = Point::new(vec![1.0, 2.0, 0.5], 1);

        let max_dist = p1.max_distance(&p2);
        assert_eq!(max_dist, 2.0); // Max of |1|, |2|, |0.5|
    }
}
