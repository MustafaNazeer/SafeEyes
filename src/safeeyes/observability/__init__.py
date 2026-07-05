"""On-device observability: structured event logging and rolling run metrics.

The live loop emits scalar telemetry only (alert transitions, face-detection
edges, and periodic latency and throughput summaries). Nothing here touches the
network or persists a raw frame, so the privacy hard rails hold.
"""
