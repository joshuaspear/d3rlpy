import pytest
import torch

from d3rlpy.models.torch.transformers import (
    GPT2,
    MLP,
    Block,
    CausalSelfAttention,
    ContinuousDecisionTransformer,
    DiscreteDecisionTransformer,
    GlobalPositionEncoding,
    SimplePositionEncoding,
)

from .model_test import DummyEncoder, check_parameter_updates


@pytest.mark.parametrize("in_size", [100])
@pytest.mark.parametrize("out_size", [100])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("batch_size", [32])
def test_causal_self_attention(
    in_size: int,
    out_size: int,
    num_heads: int,
    context_size: int,
    dropout: float,
    batch_size: int,
) -> None:
    model = CausalSelfAttention(
        in_size=in_size,
        out_size=out_size,
        num_heads=num_heads,
        context_size=context_size,
        attn_dropout=dropout,
        resid_dropout=dropout,
    )

    x = torch.rand(batch_size, context_size, in_size)
    y = model(x)

    # check shape
    assert y.shape == (batch_size, context_size, out_size)

    # check layer connections
    check_parameter_updates(model, (x,))


@pytest.mark.parametrize("in_size", [100])
@pytest.mark.parametrize("out_size", [100])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("batch_size", [32])
def test_mlp(
    in_size: int,
    out_size: int,
    context_size: int,
    dropout: float,
    batch_size: int,
) -> None:
    model = MLP(
        in_size=in_size,
        out_size=out_size,
        dropout=dropout,
        activation=torch.nn.ReLU(),
    )

    x = torch.rand(batch_size, context_size, in_size)
    y = model(x)

    # check shape
    assert y.shape == (batch_size, context_size, out_size)

    # check layer connections
    check_parameter_updates(model, (x,))


@pytest.mark.parametrize("in_size", [100])
@pytest.mark.parametrize("out_size", [100])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("batch_size", [32])
def test_block(
    in_size: int,
    out_size: int,
    num_heads: int,
    context_size: int,
    dropout: float,
    batch_size: int,
) -> None:
    model = Block(
        in_size=in_size,
        out_size=out_size,
        num_heads=num_heads,
        context_size=context_size,
        attn_dropout=dropout,
        resid_dropout=dropout,
        activation=torch.nn.ReLU(),
    )

    x = torch.rand(batch_size, context_size, in_size)
    y = model(x)

    # check shape
    assert y.shape == (batch_size, context_size, out_size)

    # check layer connections
    check_parameter_updates(model, (x,))


@pytest.mark.parametrize("max_timestep", [100])
@pytest.mark.parametrize("embed_dim", [256])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_simple_position_encoding(
    max_timestep: int,
    embed_dim: int,
    context_size: int,
    batch_size: int,
) -> None:
    model = SimplePositionEncoding(embed_dim, max_timestep)

    x = torch.randint(low=0, high=max_timestep, size=(batch_size, context_size))
    y = model(x)

    # check shape
    assert y.shape == (batch_size, context_size, embed_dim)

    # check layer connections
    check_parameter_updates(model, (x,))


@pytest.mark.parametrize("max_timestep", [3])
@pytest.mark.parametrize("embed_dim", [256])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_global_position_encoding(
    max_timestep: int,
    embed_dim: int,
    context_size: int,
    batch_size: int,
) -> None:
    model = GlobalPositionEncoding(embed_dim, max_timestep, context_size)

    x = torch.randint(
        low=0, high=max_timestep, size=(batch_size, 3 * context_size)
    )
    y = model(x)

    # check shape
    assert y.shape == (batch_size, 3 * context_size, embed_dim)


@pytest.mark.parametrize("hidden_size", [100])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("num_layers", [3])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("batch_size", [32])
def test_gpt2(
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    context_size: int,
    dropout: float,
    batch_size: int,
) -> None:
    model = GPT2(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        context_size=context_size,
        attn_dropout=dropout,
        resid_dropout=dropout,
        embed_dropout=dropout,
        activation=torch.nn.ReLU(),
    )

    x = torch.rand(batch_size, context_size, hidden_size)
    y = model(x)

    # check shape
    assert y.shape == (batch_size, context_size, hidden_size)

    # check layer connections
    check_parameter_updates(model, (x,))


@pytest.mark.parametrize("hidden_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("num_layers", [3])
@pytest.mark.parametrize("max_timestep", [20])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("batch_size", [32])
def test_continuous_decision_transformer(
    hidden_size: int,
    action_size: int,
    num_heads: int,
    num_layers: int,
    max_timestep: int,
    context_size: int,
    dropout: float,
    batch_size: int,
) -> None:
    encoder = DummyEncoder(hidden_size)

    model = ContinuousDecisionTransformer(
        encoder=encoder,
        feature_size=hidden_size,
        position_encoding=SimplePositionEncoding(hidden_size, max_timestep),
        action_size=action_size,
        num_heads=num_heads,
        num_layers=num_layers,
        context_size=context_size,
        attn_dropout=dropout,
        resid_dropout=dropout,
        embed_dropout=dropout,
        activation=torch.nn.ReLU(),
    )

    x = torch.rand(batch_size, context_size, hidden_size)
    action = torch.rand(batch_size, context_size, action_size)
    rtg = torch.rand(batch_size, context_size, 1)
    timesteps = torch.randint(0, max_timestep, size=(batch_size, context_size))
    y = model(x, action, rtg, timesteps)

    # check shape
    assert y.shape == (batch_size, context_size, action_size)

    # check layer connections
    check_parameter_updates(model, (x, action, rtg, timesteps))


@pytest.mark.parametrize("hidden_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("num_layers", [3])
@pytest.mark.parametrize("max_timestep", [20])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("batch_size", [32])
def test_discrete_decision_transformer(
    hidden_size: int,
    action_size: int,
    num_heads: int,
    num_layers: int,
    max_timestep: int,
    context_size: int,
    dropout: float,
    batch_size: int,
) -> None:
    encoder = DummyEncoder(hidden_size)

    model = DiscreteDecisionTransformer(
        encoder=encoder,
        feature_size=hidden_size,
        position_encoding=SimplePositionEncoding(hidden_size, max_timestep),
        action_size=action_size,
        num_heads=num_heads,
        num_layers=num_layers,
        context_size=context_size,
        attn_dropout=dropout,
        resid_dropout=dropout,
        embed_dropout=dropout,
        activation=torch.nn.ReLU(),
    )

    x = torch.rand(batch_size, context_size, hidden_size)
    action = torch.randint(0, action_size, size=(batch_size, context_size))
    rtg = torch.rand(batch_size, context_size, 1)
    timesteps = torch.randint(0, max_timestep, size=(batch_size, context_size))
    probs, logits = model(x, action, rtg, timesteps)

    # check shape
    assert probs.shape == (batch_size, context_size, action_size)
    assert logits.shape == (batch_size, context_size, action_size)

    # check layer connections
    check_parameter_updates(model, (x, action, rtg, timesteps))
