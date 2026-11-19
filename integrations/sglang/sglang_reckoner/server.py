# Copyright 2026 Maelie Labs
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
#
# Notice of Modification (Apache-2.0 Section 4(b)):
# Modified by Maelie Labs (2026): adapted from SGLang v0.5.22 server entrypoints
# (`sglang.srt.entrypoints.http_server`, `smg_grpc_servicer.sglang.servicer.SGLangSchedulerServicer`)
# to register Reckoner models, forward `custom_logit_processor`, and enforce front-door serving.
"""Out-of-tree SGLang token-level server launcher for Reckoner models.

Starts SGLang `0.5.22` (`--grpc-mode` or native HTTP/UDS `/generate`) as a pure
token-in / token-out generation engine behind the Go harness front door (`reckoner-harness serve`).

Key capabilities:
- Auto-registers `ReckonerConfig`, `ReckonerForCausalLM`, `Qwen35GDN2Config`, and `Qwen35GDN2ForCausalLM`
  (`SGLANG_RECKONER_AUTOLOAD=1`).
- Enables `CustomLogitProcessor` (`--enable-custom-logit-processor`) and writes the active
  `ReckonerToolLogitProcessor().to_str()` token to `/dev/shm/reckoner_sgl_processor.json`
  so `reckoner-harness serve` can attach it to token-level requests.
- Enforces front-door serving (`reckoner-harness serve`): raw `/v1/chat/completions` calls against
  SGLang return an error naming `reckoner-harness serve` unless the `--allow-raw-chat` developer
  flag is explicitly passed (off by default).
- In `--grpc-mode`, subclasses `SGLangSchedulerServicer` (`ReckonerSGLangSchedulerServicer`)
  so `GenerateRequest.custom_logit_processor` (and `custom_params["reckoner_in_engine_tools"]`)
  is forwarded into `TokenizedGenerateReqInput.custom_logit_processor` without editing SGLang source.
"""

from __future__ import annotations

import asyncio
from concurrent import futures
import json
import logging
import os
from pathlib import Path
import signal
import sys
from typing import Any, Optional

_PLUGIN_PARENT = str(Path(__file__).resolve().parents[1])
if _PLUGIN_PARENT not in sys.path:
    sys.path.insert(0, _PLUGIN_PARENT)
_cur_pypath = os.environ.get("PYTHONPATH", "")
if _PLUGIN_PARENT not in _cur_pypath.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{_PLUGIN_PARENT}{os.pathsep}{_cur_pypath}" if _cur_pypath else _PLUGIN_PARENT
os.environ.setdefault("SGLANG_RECKONER_AUTOLOAD", "1")
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

if __package__:
    from . import (
        _maybe_canonicalize_seq,
        install_front_door_chat_guard,
        is_reckoner_model_active,
        register_reckoner_with_sglang,
        set_active_model_arch,
    )
    from .tool_processor import ReckonerToolLogitProcessor
else:
    from sglang_reckoner import (  # type: ignore[no-redef]
        _maybe_canonicalize_seq,
        install_front_door_chat_guard,
        is_reckoner_model_active,
        register_reckoner_with_sglang,
        set_active_model_arch,
    )
    from sglang_reckoner.tool_processor import ReckonerToolLogitProcessor  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

DEFAULT_PROCESSOR_INFO_PATH = "/dev/shm/reckoner_sgl_processor.json"


def parse_reckoner_cli_flags(argv: list[str]) -> tuple[list[str], bool]:
    """Extract Reckoner-specific CLI flags (`--allow-raw-chat`, `--harness-bin`) and return `(remaining_args, allow_raw_chat)`."""
    remaining: list[str] = []
    allow_raw_chat = False
    i = 0
    n = len(argv)
    while i < n:
        arg = argv[i]
        if arg == "--allow-raw-chat":
            allow_raw_chat = True
        elif arg == "--harness-bin" and i + 1 < n:
            os.environ["RECKONER_HARNESS_BIN"] = argv[i + 1]
            i += 1
        elif arg.startswith("--harness-bin="):
            os.environ["RECKONER_HARNESS_BIN"] = arg.split("=", 1)[1]
        else:
            remaining.append(arg)
        i += 1
    return remaining, allow_raw_chat


def _detect_and_set_model_arch(model_path: Optional[str]) -> Optional[str]:
    """Inspect `<model_path>/config.json` and record the model's architecture name."""
    if not model_path:
        return None
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            archs = cfg.get("architectures") or []
            if archs and isinstance(archs[0], str):
                set_active_model_arch(archs[0])
                return archs[0]
            if cfg.get("model_type") == "reckoner":
                set_active_model_arch("ReckonerForCausalLM")
                return "ReckonerForCausalLM"
            if cfg.get("model_type") == "qwen35_gdn2":
                set_active_model_arch("Qwen35GDN2ForCausalLM")
                return "Qwen35GDN2ForCausalLM"
        except Exception:
            pass
    return None


def write_processor_info(path: str = DEFAULT_PROCESSOR_INFO_PATH) -> str:
    """Serializes `ReckonerToolLogitProcessor().to_str()` to JSON for the Go front door."""
    proc_str = ReckonerToolLogitProcessor().to_str()
    mode_str = ReckonerToolLogitProcessor.log_active_path()
    payload = {
        "custom_logit_processor": proc_str,
        "class_name": f"{ReckonerToolLogitProcessor.__module__}.{ReckonerToolLogitProcessor.__qualname__}",
        "inflight_extend": ReckonerToolLogitProcessor.supports_inflight_extend(),
        "observation_injection_mode": mode_str,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)
    return proc_str


async def serve_grpc_reckoner(server_args: Any) -> None:
    """Runs SGLang's gRPC server with `ReckonerSGLangSchedulerServicer` forwarding `custom_logit_processor`."""
    import grpc
    from grpc_health.v1 import health_pb2_grpc
    from grpc_reflection.v1alpha import reflection
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import PortArgs
    from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc
    from smg_grpc_servicer.hostport import host_port
    from smg_grpc_servicer.sglang.config_view import EngineConfigView
    from smg_grpc_servicer.sglang.health_servicer import SGLangHealthServicer
    from smg_grpc_servicer.sglang.request_manager import GrpcRequestManager
    from smg_grpc_servicer.sglang.scheduler_launcher import (
        launch_scheduler_process_only,
        terminate_scheduler_processes,
    )
    from smg_grpc_servicer.sglang.servicer import SGLangSchedulerServicer

    register_reckoner_with_sglang()
    _detect_and_set_model_arch(server_args.model_path)
    if not server_args.tokenizer_path:
        server_args.tokenizer_path = server_args.model_path
    if server_args.tokenizer_path:
        os.environ["RECKONER_TOKENIZER_PATH"] = server_args.tokenizer_path
    server_args.enable_custom_logit_processor = True
    server_args.disable_overlap_schedule = True
    if not getattr(server_args, "cuda_graph_max_bs", None):
        server_args.cuda_graph_max_bs = 32
    default_proc_str = write_processor_info()

    class ReckonerSGLangSchedulerServicer(SGLangSchedulerServicer):
        """Subclass of `SGLangSchedulerServicer` forwarding `custom_logit_processor`."""

        def _convert_generate_request(
            self,
            grpc_req: sglang_scheduler_pb2.GenerateRequest,
        ) -> TokenizedGenerateReqInput:
            req = super()._convert_generate_request(grpc_req)
            if (
                getattr(req, "input_ids", None)
                and os.environ.get("SGLANG_RECKONER_CANONICALIZE_TURNS", "0") == "1"
                and is_reckoner_model_active()
            ):
                req.input_ids = _maybe_canonicalize_seq(
                    list(req.input_ids), rid=getattr(req, "rid", None)
                )
            custom_lp = grpc_req.custom_logit_processor
            if not custom_lp and isinstance(req.sampling_params.custom_params, dict):
                if req.sampling_params.custom_params.get("reckoner_in_engine_tools"):
                    custom_lp = default_proc_str
            if custom_lp:
                req.custom_logit_processor = custom_lp
            return req

    publish(server_args, role="tokenizer")
    port_args = PortArgs.init_new(server_args)
    request_manager = GrpcRequestManager(server_args=server_args, port_args=port_args)
    scheduler_info, _, scheduler_procs = launch_scheduler_process_only(
        server_args=server_args,
        port_args=port_args,
    )

    model_config = ModelConfig.from_server_args(server_args)
    facts = EngineConfigView(server_args, model_config).model
    if facts.architectures:
        set_active_model_arch(facts.architectures[0])
    elif facts.model_type == "reckoner":
        set_active_model_arch("ReckonerForCausalLM")
    elif facts.model_type == "qwen35_gdn2":
        set_active_model_arch("Qwen35GDN2ForCausalLM")
    model_info = {
        "model_name": server_args.model_path,
        "max_context_length": scheduler_info.get(
            "max_total_num_tokens", server_args.context_length or 8192
        ),
        "vocab_size": scheduler_info.get("vocab_size", 36864),
        "supports_vision": False,
        "model_type": facts.model_type or "reckoner",
        "architectures": facts.architectures or ["ReckonerForCausalLM"],
        "max_req_input_len": scheduler_info.get("max_req_input_len", 8192),
        "eos_token_ids": scheduler_info.get("eos_token_ids", [4, 0]),
        "pad_token_id": scheduler_info.get("pad_token_id", 1),
        "bos_token_id": scheduler_info.get("bos_token_id", 0),
        "id2label_json": "",
        "num_labels": 0,
    }

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )
    health_servicer = SGLangHealthServicer(
        request_manager=request_manager,
        scheduler_info=scheduler_info,
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    servicer = ReckonerSGLangSchedulerServicer(
        request_manager=request_manager,
        server_args=server_args,
        model_config=model_config,
        model_info=model_info,
        scheduler_info=scheduler_info,
        health_servicer=health_servicer,
    )
    sglang_scheduler_pb2_grpc.add_SglangSchedulerServicer_to_server(servicer, server)
    service_names = (
        sglang_scheduler_pb2.DESCRIPTOR.services_by_name["SglangScheduler"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = host_port(server_args.host, server_args.port)
    server.add_insecure_port(listen_addr)
    await server.start()
    health_servicer.set_serving()
    logger.info("Reckoner gRPC token server listening on %s", listen_addr)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    finally:
        health_servicer.set_not_serving()
        await server.stop(grace=2.0)
        terminate_scheduler_processes(scheduler_procs)
        await request_manager.shutdown()


def main(argv: Optional[list[str]] = None) -> None:
    """Entrypoint for launching the SGLang token-level server (`--grpc-mode` or HTTP `/generate`)."""
    os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    os.environ.setdefault("SGLANG_RECKONER_FUSED_GATES", "1")
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args

    raw_args = list(sys.argv[1:] if argv is None else argv)
    raw_args, allow_raw_chat = parse_reckoner_cli_flags(raw_args)
    if allow_raw_chat:
        os.environ["SGLANG_RECKONER_ALLOW_RAW_CHAT"] = "1"
    else:
        os.environ.setdefault("SGLANG_RECKONER_ALLOW_RAW_CHAT", "0")

    register_reckoner_with_sglang()
    install_front_door_chat_guard()
    write_processor_info()

    if "--enable-custom-logit-processor" not in raw_args:
        raw_args.append("--enable-custom-logit-processor")
    if "--trust-remote-code" not in raw_args:
        raw_args.append("--trust-remote-code")
    if (
        "--disable-overlap-schedule" not in raw_args
        and os.environ.get("SGLANG_RECKONER_ENABLE_OVERLAP", "0") != "1"
    ):
        raw_args.append("--disable-overlap-schedule")
    if (
        "--cuda-graph-max-bs-decode" not in raw_args
        and "--disable-cuda-graph" not in raw_args
    ):
        raw_args.extend(["--cuda-graph-max-bs-decode", "32"])

    server_args = prepare_server_args(raw_args)
    _detect_and_set_model_arch(server_args.model_path)
    if not server_args.tokenizer_path:
        server_args.tokenizer_path = server_args.model_path
    if server_args.tokenizer_path:
        os.environ["RECKONER_TOKENIZER_PATH"] = server_args.tokenizer_path
    if server_args.grpc_mode:
        asyncio.run(serve_grpc_reckoner(server_args))
    else:
        launch_server(server_args)


if __name__ == "__main__":
    main()
