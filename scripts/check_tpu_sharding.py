#!/usr/bin/env python3
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree
import numpy as np

import hfax
from kauldron import kd
import optax
from kauldron.train.train_step import partial_loader


def main() -> None:
    rng_streams = kd.train.RngStreams(
        (
            kd.train.RngStream(
                name="dropout", init=True, train=True, eval=False, per_step=True
            ),
        ),
        seed=42,
    )

    trainstep = kd.train.train_step.TrainStep(
        model=hfax.nn.Gemma3_4B(tokens="batch.input"),
        optimizer=optax.adafactor(learning_rate=1e-3),
        rng_streams=rng_streams,
        sharding=kd.train.train_step.sharding_lib.ShardingStrategy(),
        init_transform=partial_loader.NoopTransform(),
        aux=None,
    )

    elem_spec = {
        "input": jax.ShapeDtypeStruct((8, 1024), jnp.int32),
        "target": jax.ShapeDtypeStruct((8, 1024, 1), jnp.int32),
        "loss_mask": jax.ShapeDtypeStruct((8, 1024, 1), jnp.int32),
    }

    state = trainstep.init(elem_spec=elem_spec)

    print(f"local_device_count: {jax.local_device_count()}")
    param_shardings = {str(x.sharding) for x in jax.tree.leaves(state.params)}
    opt_shardings = {str(x.sharding) for x in jax.tree.leaves(state.opt_state)}
    print("params shardings:")
    for sh in sorted(param_shardings):
        print("  ", sh)
    print("optimizer shardings:")
    for sh in sorted(opt_shardings):
        print("  ", sh)

    mesh = jax.sharding.Mesh(
        np.array(jax.devices(), dtype=object), ("devices",)
    )
    print("mesh:", mesh)


if __name__ == "__main__":
    main()
