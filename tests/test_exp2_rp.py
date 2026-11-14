"""Penalty-rollout variant parsing and consensus shard interleaving."""
from votedistill.exp2.runs import OFFICIAL_CLIP, objective_of, parse_name, rollout_pen_of, style_of


def test_rp_variants():
    assert parse_name("L3-vote__copsd__exact__rp__s42")[4] == OFFICIAL_CLIP
    assert rollout_pen_of("L3-vote__copsd__exact__rp__s42") == 1.1
    assert parse_name("L3-vote__copsd__exact__rp-noclip__s43")[4] is None
    assert rollout_pen_of("L3-vote__copsd__exact__noclip__s43") == 1.0
    assert rollout_pen_of("L3-vote__copsd__exact__s42") == 1.0
    assert objective_of("L0-same__copsd__exact__rp-hot1.5__s42") == ("copsd", 1.5)
    assert style_of("L3-vote__copsd__exact__rp__s42") == "copsd"
    assert parse_name("selfsft-vote__copsd__exact__rp__s42")[0] == "selfsft-vote"


def test_shard_interleave():
    items = list(range(23))
    k = 5
    shards = [items[i::k] for i in range(k)]
    assert [shards[j % k][j // k] for j in range(sum(map(len, shards)))] == items
