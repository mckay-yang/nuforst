use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
#[derive(Deserialize)]
pub struct SceneSources {
    pub sources: Vec<String>,
    pub tiles: Vec<String>,
}
impl SceneSources {
    pub fn select(&self, epochs: &[f64], valid: &[bool]) -> Vec<usize> {
        let mut by_tile: BTreeMap<&str, BTreeMap<u64, usize>> = BTreeMap::new();
        for i in 0..epochs.len() {
            if !valid[i] {
                continue;
            }
            let dates = by_tile.entry(&self.tiles[i]).or_default();
            let old = dates.entry(epochs[i].to_bits()).or_insert(i);
            if (&self.sources[i], i) < (&self.sources[*old], *old) {
                *old = i;
            }
        }
        let mut best = Vec::new();
        for dates in by_tile.values() {
            if dates.len() > best.len() {
                best = dates.values().copied().collect();
            }
        }
        best.sort_by(|&a, &b| epochs[a].total_cmp(&epochs[b]).then(a.cmp(&b)));
        debug_assert_eq!(
            best.len(),
            best.iter()
                .map(|&i| epochs[i].to_bits())
                .collect::<BTreeSet<_>>()
                .len()
        );
        best
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_is_availability_only_with_deterministic_ties() {
        let s = SceneSources {
            sources: vec!["z".into(), "b".into(), "a".into(), "c".into(), "d".into()],
            tiles: vec!["A".into(), "A".into(), "A".into(), "B".into(), "B".into()],
        };
        assert_eq!(s.select(&[0., 1., 1., 0., 2.], &[true; 5]), vec![0, 2]);
        assert_eq!(
            s.select(&[0., 1., 1., 0., 2.], &[false, true, true, true, true]),
            vec![3, 4]
        );
    }
}
