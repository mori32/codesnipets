//! A* path prefixes in a reusable arena, not a cloned Vec for every branch.
//! The heap preserves the reference bound/lexicographic ordering, including ties.
use std::cmp::Ordering;

use crate::optimized::EdgeScores;
use crate::{Lattice, Model, Path, Result};

#[derive(Debug)]
struct Prefix {
    parent: usize,
    node: usize,
    depth: usize,
    score: f64,
}
#[derive(Debug)]
struct Item {
    bound: f64,
    prefix: usize,
}

#[derive(Default)]
pub(crate) struct Workspace {
    scores: EdgeScores,
    suffix: Vec<f64>,
    prefixes: Vec<Prefix>,
    heap: Vec<Item>,
}

impl Workspace {
    fn path_cmp(&self, mut a: usize, mut b: usize) -> Ordering {
        let a_depth = self.prefixes[a].depth;
        let b_depth = self.prefixes[b].depth;
        while self.prefixes[a].depth > self.prefixes[b].depth {
            a = self.prefixes[a].parent;
        }
        while self.prefixes[b].depth > self.prefixes[a].depth {
            b = self.prefixes[b].parent;
        }
        if a == b {
            return a_depth.cmp(&b_depth);
        }
        // Prefixes form a tree: the children below the common ancestor contain
        // the first different node in the two lexicographic path sequences.
        while self.prefixes[a].parent != self.prefixes[b].parent {
            a = self.prefixes[a].parent;
            b = self.prefixes[b].parent;
        }
        self.prefixes[a].node.cmp(&self.prefixes[b].node)
    }
    fn cmp(&self, a: usize, b: usize) -> Ordering {
        let a = &self.heap[a];
        let b = &self.heap[b];
        a.bound
            .total_cmp(&b.bound)
            .then_with(|| self.path_cmp(b.prefix, a.prefix))
    }
    fn push(&mut self, item: Item) {
        self.heap.push(item);
        let mut child = self.heap.len() - 1;
        while child != 0 {
            let parent = (child - 1) / 2;
            if self.cmp(child, parent) != Ordering::Greater {
                break;
            }
            self.heap.swap(child, parent);
            child = parent;
        }
    }
    fn pop(&mut self) -> Option<Item> {
        if self.heap.is_empty() {
            return None;
        }
        let result = self.heap.swap_remove(0);
        let mut parent = 0_usize;
        while let Some(left) = parent
            .checked_mul(2_usize)
            .and_then(|n| n.checked_add(1))
            .filter(|&n| n < self.heap.len())
        {
            let right = left + 1;
            let child = if right < self.heap.len() && self.cmp(right, left) == Ordering::Greater {
                right
            } else {
                left
            };
            if self.cmp(child, parent) != Ordering::Greater {
                break;
            }
            self.heap.swap(child, parent);
            parent = child;
        }
        Some(result)
    }
    fn path(&self, mut index: usize) -> Path {
        let score = self.prefixes[index].score;
        let mut nodes = Vec::with_capacity(self.prefixes[index].depth);
        loop {
            nodes.push(self.prefixes[index].node);
            if index == 0 {
                break;
            }
            index = self.prefixes[index].parent;
        }
        nodes.reverse();
        Path { nodes, score }
    }

    pub(crate) fn nbest(
        &mut self,
        model: &Model,
        graph: &Lattice,
        count: usize,
        limit: usize,
    ) -> Result<Vec<Path>> {
        if count == 0 {
            return Err("N-best size must be positive".into());
        }
        self.scores.fill(model, graph);
        let last = graph.nodes.len() - 1;
        self.suffix.resize(graph.nodes.len(), f64::NEG_INFINITY);
        self.suffix.fill(f64::NEG_INFINITY);
        self.suffix[last] = 0.0;
        for node in (0..last).rev() {
            for &e in &graph.outgoing[node] {
                self.suffix[node] =
                    self.suffix[node].max(self.scores.values[e] + self.suffix[graph.edges[e].to]);
            }
        }
        if !self.suffix[0].is_finite() {
            return Err("no finite Viterbi path".into());
        }
        self.prefixes.clear();
        self.heap.clear();
        self.prefixes.push(Prefix {
            parent: 0,
            node: 0,
            depth: 1,
            score: 0.0,
        });
        self.push(Item {
            bound: self.suffix[0],
            prefix: 0,
        });
        let mut results = Vec::new();
        let mut expansions = 0;
        while let Some(state) = self.pop() {
            let node = self.prefixes[state.prefix].node;
            if node == last {
                results.push(self.path(state.prefix));
                if results.len() == count {
                    break;
                }
                continue;
            }
            expansions += 1;
            if expansions > limit {
                return Err("A* expansion budget exceeded (not silently beam-pruned)".into());
            }
            for &e in &graph.outgoing[node] {
                let target = graph.edges[e].to;
                if !self.suffix[target].is_finite() {
                    continue;
                }
                let score = self.prefixes[state.prefix].score + self.scores.values[e];
                let index = self.prefixes.len();
                self.prefixes.push(Prefix {
                    parent: state.prefix,
                    node: target,
                    depth: self.prefixes[state.prefix].depth + 1,
                    score,
                });
                self.push(Item {
                    bound: score + self.suffix[target],
                    prefix: index,
                });
            }
        }
        Ok(results)
    }
}
