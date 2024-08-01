use std::{
    collections::BTreeMap,
    ops::{Index, IndexMut, Not},
};

use IndexConstraint as IC;
use ReducedIndexConstraint as RIC;

#[derive(Copy, Clone, Debug)]
enum Delim {
    Start,
    End,
}

impl Not for Delim {
    type Output = Delim;

    fn not(self) -> Self::Output {
        return match self {
            Delim::Start => Delim::End,
            Delim::End => Delim::Start,
        };
    }
}

#[derive(Default, Debug)]
/// Tool that reduces constraints by converting them to delimiters and regenerating constraints from the delimiters.
struct DelimReducer(BTreeMap<usize, Vec<Delim>>);

impl DelimReducer {
    /// Helper to aid in folding together instances of SortedIndexDelims.
    fn concat(&mut self, rhs: DelimReducer) -> &mut DelimReducer {
        for (idx, bounds) in rhs.0.into_iter() {
            self.0
                .entry(idx)
                .and_modify(|curr| curr.extend_from_slice(&bounds))
                .or_insert(bounds);
        }
        return self;
    }

    /// Inserts a single delimiter.
    fn insert(&mut self, idx: usize, delim: Delim) {
        self.0
            .entry(idx)
            .and_modify(|curr| curr.push(delim))
            .or_insert(vec![delim]);
    }

    // Insers the delimiters corresponding to a [TensorIndexConstraint::Range].
    fn insert_start_end(&mut self, start: usize, end: usize) {
        self.insert(start, Delim::Start);
        self.insert(end, Delim::End);
    }

    /// Inserts the delimiters corresponding to a [TensorIndexConstaint::Single].
    fn insert_unit(&mut self, idx: usize) {
        self.insert_start_end(idx, idx + 1);
    }

    /// Constructs a SortedIndexDelims from a set of constraints.
    fn construct(value: ReducedIndexConstraint) -> DelimReducer {
        let mut delims = DelimReducer::default();
        match value {
            RIC::Range(start, len) => delims.insert_start_end(start, start + len),
            RIC::Single(idx) => delims.insert_unit(idx),
            RIC::Multi(idx_vec) => _ = idx_vec.into_iter().map(|idx| delims.insert_unit(idx)),
            RIC::Union(cons_vec, _) => {
                _ = cons_vec
                    .into_iter()
                    // Reduce all children
                    .map(|cons| cons.reduce())
                    // Flatten any child unions
                    .map(|cons| match cons {
                        RIC::Union(cons_vec, _) => cons_vec,
                        cons => vec![cons],
                    })
                    .flatten()
                    // Construct delimiters for each child element
                    .map(|cons| DelimReducer::construct(cons))
                    // Combine delimiters of all child elements
                    .fold(&mut delims, DelimReducer::concat)
            }
            RIC::From(idx) => delims.insert(idx, Delim::Start),
            RIC::Null => {}
        };
        return delims;
    }

    /// Reduces the bounds to the most simplified set of [TensorIndexConstraint]s.
    fn simplified_constraint(&self) -> ReducedIndexConstraint {
        let mut cons_vec = Vec::new();

        let mut open_stack_level = 0;
        let mut current_start = None;
        let mut current_multi: Vec<usize> = Vec::new();

        for (idx, delims) in self.0.iter() {
            // Determine whether in included section or not.
            for delim in delims {
                match delim {
                    Delim::Start => open_stack_level += 1,
                    Delim::End => open_stack_level -= 1,
                }
            }

            let mut to_push = None;

            // Either start or end a range when appropriate.
            match current_start {
                None if open_stack_level > 0 => current_start = Some(idx),
                Some(start_idx) if open_stack_level <= 0 => {
                    // This can only reduce to an instance of Single
                    to_push = Some(RIC::Range(*start_idx, idx - start_idx).reduce());
                    current_start = None;
                }
                _ => {}
            }

            // Continue if nothing to push.
            let Some(to_push) = to_push else {
                continue;
            };

            if let RIC::Single(idx) = to_push {
                current_multi.push(idx);
                continue;
            }

            if current_multi.len() > 0 {
                cons_vec.push(RIC::Multi(current_multi.drain(..).collect()).reduce());
            }

            cons_vec.push(to_push);
        }

        if current_multi.len() > 0 {
            cons_vec.push(RIC::Multi(current_multi.drain(..).collect()).reduce());
        }

        // Push a From contraint if open after last delim.
        if let Some(start_idx) = current_start {
            cons_vec.push(RIC::From(*start_idx));
        }

        return RIC::Union(cons_vec, true).reduce();
    }

    /// Constructs the delimiters and then simplifies the constraints.
    pub fn reduce(value: ReducedIndexConstraint) -> ReducedIndexConstraint {
        return DelimReducer::construct(value).simplified_constraint();
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum ReducedIndexConstraint {
    Single(usize),
    Multi(Vec<usize>),
    Range(usize, usize),
    From(usize),
    Union(Vec<ReducedIndexConstraint>, bool),
    Null,
}

impl ReducedIndexConstraint {
    fn reduce_multi(idx_vec: Vec<usize>) -> ReducedIndexConstraint {
        return match idx_vec[..] {
            [] => RIC::Null,
            [a] => RIC::Single(a),
            _ => RIC::Multi(idx_vec),
        };
    }

    fn reduce_range(start: usize, len: usize) -> ReducedIndexConstraint {
        return match len {
            0 => RIC::Null,
            1 => RIC::Single(start),
            _ => RIC::Range(start, len),
        };
    }

    fn reduce_union(
        mut cons_vec: Vec<ReducedIndexConstraint>,
        delim_reduced: bool,
    ) -> ReducedIndexConstraint {
        match (&cons_vec[..], delim_reduced) {
            ([], _) => RIC::Null,
            ([_], _) => cons_vec.pop().unwrap().reduce(),
            (_, true) => RIC::Union(cons_vec, true),
            (_, false) => DelimReducer::reduce(RIC::Union(cons_vec, false)),
        }
    }

    fn reduce(self) -> ReducedIndexConstraint {
        match self {
            RIC::Multi(idx_vec) => RIC::reduce_multi(idx_vec),
            RIC::Range(start, len) => RIC::reduce_range(start, len),
            RIC::Union(cons_vec, delim_reduced) => Self::reduce_union(cons_vec, delim_reduced),
            RIC::Single(_) | RIC::From(_) | RIC::Null => self,
        }
    }

    fn len(&self) -> Option<usize> {
        return match self {
            RIC::Single(_) => Some(1),
            RIC::Multi(idx_vec) => Some(idx_vec.len()),
            RIC::Range(_, len) => Some(*len),
            RIC::From(_) => None,
            RIC::Union(cons_vec, _) => cons_vec
                .iter()
                .map(|cons| cons.len())
                .try_fold(0usize, |acc, x| acc.checked_add(x?)),
            RIC::Null => Some(0),
        };
    }

    fn index_union(cons_vec: &Vec<ReducedIndexConstraint>, mut src_idx: usize) -> Option<usize> {
        for cons in cons_vec {
            let len = cons.len();
            let Some(len) = len else {
                return Some(src_idx);
            };
            if len < src_idx {
                src_idx -= len;
                continue;
            }
            return cons.try_index(src_idx);
        }
        return None;
    }

    fn try_index(&self, src_idx: usize) -> Option<usize> {
        return match (src_idx, self) {
            (0, RIC::Single(idx)) => Some(*idx),
            (_, RIC::Single(_)) => None,
            (src_idx, RIC::Multi(idx_vec)) if src_idx < idx_vec.len() => Some(idx_vec[src_idx]),
            (_, RIC::Multi(_)) => None,
            (src_idx, RIC::Range(start, len)) if src_idx < *len => Some(start + src_idx),
            (_, RIC::Range(_, _)) => None,
            (src_idx, RIC::From(start)) => Some(start + src_idx),
            (src_idx, RIC::Union(cons_vec, _)) => Self::index_union(cons_vec, src_idx),
            (_, RIC::Null) => None,
        };
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
/// Constraint for indexing one dimension of a tensor.
///
/// This allows for highly customizable views of tensors.
pub enum IndexConstraint {
    /// Index a single index within the dimension.
    Single(usize),
    /// Index a range of indices within the dimenions.
    Range(usize, usize),
    /// Range until end.
    From(usize),
    /// Index the union of multiple constraints.
    Union(Vec<IndexConstraint>),
    /// Entirety of possible range.
    All,
    /// None of the range.
    Null,
}

impl IndexConstraint {
    fn reduce(self) -> ReducedIndexConstraint {
        let ric = match self {
            IC::Single(idx) => RIC::Single(idx),
            IC::Range(idx_from, idx_to) => RIC::Range(idx_from, idx_to - idx_from),
            IC::From(idx) => RIC::From(idx),
            IC::Union(cons_vec) => {
                RIC::Union(cons_vec.into_iter().map(|ic| ic.reduce()).collect(), false)
            }
            IC::All => RIC::From(0),
            IC::Null => RIC::Null,
        };
        return ric.reduce();
    }

    pub fn compile(self) -> IndexMask {
        return IndexMask::from(self.reduce());
    }
}

pub struct IndexMask {
    cons: ReducedIndexConstraint,
    len: Option<usize>,
}

impl IndexMask {
    pub fn mask<'a, T>(&'a self, to_mask: &'a T) -> MaskedIndex<'a, &T>
    where
        T: Index<usize>,
    {
        return MaskedIndex {
            mask: &self,
            data: to_mask,
        };
    }

    pub fn mask_mut<'a, T>(&'a self, to_mask: &'a mut T) -> MaskedIndex<'a, &mut T>
    where
        T: IndexMut<usize>,
    {
        return MaskedIndex {
            mask: &self,
            data: to_mask,
        };
    }

    pub fn try_index(&self, src_idx: usize) -> Option<usize> {
        return match self.len {
            Some(len) if len > src_idx => None,
            _ => self.cons.try_index(src_idx),
        };
    }
}

impl From<ReducedIndexConstraint> for IndexMask {
    fn from(value: ReducedIndexConstraint) -> Self {
        return IndexMask {
            len: value.len(),
            cons: value,
        };
    }
}

pub struct MaskedIndex<'a, T> {
    mask: &'a IndexMask,
    data: T,
}

impl<'a, T> MaskedIndex<'a, T>
where
    T: Index<usize>,
{
    pub fn try_index(&self, src_index: usize) -> Option<&T::Output> {
        let index = self.mask.try_index(src_index)?;
        return Some(&self.data[index]);
    }
}

impl<'a, T> MaskedIndex<'a, T>
where
    T: IndexMut<usize>,
{
    pub fn try_index_mut(&mut self, src_index: usize) -> Option<&mut T::Output> {
        let index = self.mask.try_index(src_index)?;
        return Some(&mut self.data[index]);
    }
}

impl<'a, T> Index<usize> for MaskedIndex<'a, T>
where
    T: Index<usize>,
{
    type Output = T::Output;

    fn index(&self, index: usize) -> &Self::Output {
        return self.try_index(index).expect("Index out of bounds.");
    }
}

impl<'a, T> IndexMut<usize> for MaskedIndex<'a, T>
where
    T: IndexMut<usize>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return self.try_index_mut(index).expect("Index out of bounds.");
    }
}

#[cfg(test)]
mod tests {
    use super::IndexConstraint as IC;
    use super::ReducedIndexConstraint as RIC;
    use test_case::test_case;

    #[test_case(0; "Overlapping index ranges.")]
    #[test_case(1; "Adjacent single indicies.")]
    #[test_case(2; "Overlapping single indices and ranges.")]
    #[test_case(3; "Overlapping single indices and ranges, nested in unions.")]
    fn test_reductions(idx: usize) {
        let pairs = [
            (
                IC::Union(vec![IC::Range(2, 4), IC::Range(3, 5)]),
                RIC::Range(2, 3),
            ),
            (
                IC::Union(vec![IC::Single(4), IC::Single(5), IC::Single(6)]),
                RIC::Range(4, 3),
            ),
            (
                IC::Union(vec![
                    IC::Single(2),
                    IC::Single(5),
                    IC::Single(7),
                    IC::Range(6, 9),
                    IC::Single(20),
                ]),
                RIC::Union(
                    vec![RIC::Single(2), RIC::Range(5, 4), RIC::Single(20)],
                    true,
                ),
            ),
            (
                IC::Union(vec![
                    IC::Union(vec![IC::Single(2), IC::Range(6, 9), IC::Single(20)]),
                    IC::Union(vec![IC::Single(5)]),
                    IC::Single(7),
                ]),
                RIC::Union(
                    vec![RIC::Single(2), RIC::Range(5, 4), RIC::Single(20)],
                    true,
                ),
            ),
        ];

        let (unreduced, reduced) = pairs[idx].clone();
        assert_eq!(unreduced.reduce(), reduced);
    }
}
