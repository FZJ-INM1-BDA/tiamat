.. title:: FAQ - tiamat

==========================
Frequently Asked Questions
==========================

| **Q: Will tiamat always be as fast as precomputed pyramids?**
| A: No. Precomputed pyramids are optimized for read-heavy workloads and can beat on-the-fly pipelines on raw latency. tiamat trades a modest runtime overhead for flexibility and storage efficiency. We suggest using tiamat for exploration, dynamic transforms and versioned pipelines - and converting to precomputed outputs for finalized, heavily accessed datasets.

| **Q: How do I decide whether to use tiamat or convert my data?**
| A: Consider (a) access frequency, (b) data volatility, (c) storage costs, and (d) whether you need on-demand transforms. If queries are sparse or transforms change often, tiamat is an excellent fit. If millions of users will repeatedly hit the same tiles, precompute.

| **Q: Is tiamat secure for public serving?**
| A: tiamat itself is a data layer; security depends on how you deploy it. Run behind an authenticated reverse proxy, use HTTPS, and limit public write access. When exposing model inference, ensure you control resource quotas.

| **Q: Can I do GPU inference with tiamat?**
| A: Yes. Transformers may call GPU code (PyTorch, ONNX runtime with CUDA). In cluster setups, consider remote inference: the transformer can forward tiles to a GPU microservice and stitch results back.
