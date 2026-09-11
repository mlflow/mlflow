# PR #24999 — 2026-09-10 Review Triage

范围：只包含 2026-09-10 当天由 **mprahl** 与 **Copilot** 提出的、尚未 resolved 的 inline review comments。此前评论不在本次范围内。

## 总结

- 共 **11** 条未解决评论：Copilot 5 条，mprahl 6 条。
- 本次 review 的顶层反馈只有 `Changes requested`，没有额外技术要求。
- 必须优先修复的根节点是：空 scope 的 presence 语义（fail-open）和未使用新字段时的第三方 store 向后兼容性。

## 汇总表

| 组  | Comment ID                                                                                                        | 文件                                                 | 问题摘要                                                                                                                          | 优先级        |
| --- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| A   | [r3982626123](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982626123)                                 | `mlflow/server/handlers.py:1095`                     | `_raw_request_has_field()` 仅识别 snake_case；`experimentIds: []` 会被解析进 proto，却被当作字段缺席，空 scope 退化为无过滤查询。 | P0            |
| B   | [r3982626162](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982626162)（含 Copilot suppressed 同类项） | `mlflow/server/handlers.py:4203`                     | legacy 请求仍传入 `experiment_ids=None`，使用旧方法签名的 entry-point tracking store 会抛 `TypeError`。                           | P0            |
| C   | [r3982626196](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982626196)                                 | `mlflow/server/handlers.py:5885`                     | 代码拒绝同时给出 `experiment_id` 和 `experiment_ids`，但 PR 描述仍称二者会 merge；公开 API 合约不一致。                           | P1            |
| D   | [r3982626251](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982626251)                                 | `mlflow/store/tracking/sqlalchemy_store.py:5983`     | `experiment_ids` 含重复值且跨 chunk 时，同一 trace/info 可被重复返回。                                                            | P1            |
| E   | [r3982626297](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982626297)                                 | `mlflow/store/tracking/databricks_rest_store.py:332` | appendix 仍称 Databricks 静默丢弃 scope；现实现已变为显式 fail-closed 400。                                                       | P2            |
| F   | [r3982630748](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982630748)                                 | `mlflow/store/tracking/sqlalchemy_store.py:2998`     | 建议让 `list_active_experiment_ids(None)` 枚举全部 active IDs，以移除 `_search_active_experiment_ids`。                           | P2 / 设计选择 |
| G   | [r3982636698](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982636698)                                 | `mlflow/server/handlers.py:5895`                     | 质疑预先 `_validate_experiment_id` 是否重复：`list_active_experiment_ids` 会过滤无效/非 active ID。                               | P2            |
| H   | [r3982656825](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982656825)                                 | `mlflow/store/tracking/sqlalchemy_store.py:3015`     | `list_active_experiment_ids` 是否需显式 `ORDER BY` 来保证返回顺序。                                                               | P2            |
| I   | [r3982668349](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982668349)                                 | `mlflow/server/handlers.py:4195`                     | `batch_get_trace_infos` 的 Databricks pre-check 似乎不属于本 PR，建议移除。                                                       | P2            |
| J   | [r3982685592](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982685592)                                 | `mlflow/store/tracking/sqlalchemy_store.py:5972`     | docstring 不应限定为 authorization；这是通用的 experiment ID filter 能力。                                                        | P3            |
| K   | [r3982698325](https://github.com/mlflow/mlflow/pull/24999#discussion_r3982698325)                                 | `mlflow/store/tracking/sqlalchemy_store.py:5990`     | 建议把 chunking 下沉到 `_filter_experiment_ids`，减少调用方的重复逻辑。                                                           | P2 / 设计选择 |

## 推荐处理顺序（root nodes 优先）

```text
A. raw request presence：同时识别 proto field.name 与 field.json_name
   └─> 补 `experimentIds: []` 的 camelCase deny-all 回归测试

B. legacy store 兼容：字段缺席时保持旧调用签名
   └─> 同时修改 batch_get_traces 与 batch_get_trace_infos
   └─> 补旧签名自定义 store 的回归测试

D. scope ID 规范化：解析后、chunk 前保序去重
   └─> 覆盖两个 batch API 的跨 chunk 重复-ID 场景

C. scorer API 合约
   └─> 保留当前两个字段互斥的实现，更新 PR 描述的 "merges" 表述

E / I / J. 文档、无关代码、措辞收尾

F / G / H / K. 与 mprahl 确认设计取舍后再改
```

## 分组上下文与建议动作

### A — `experiment_ids` presence 是安全根节点

protobuf `repeated` 字段无法从 message 本身区分"未传"和"显式传空列表"。当前 helper 通过原始 JSON 判断 presence，但只检查 `experiment_ids`，忽略 protobuf JSON 支持的 `experimentIds`。后者传空数组时会造成 deny-all scope 变为 `None`，从而无过滤查询。

建议：让 helper 基于 descriptor 同时识别 `field.name` 和 `field.json_name`，并为两个 batch handler 添加真实 Flask request 的 camelCase 空数组回归测试。

### B — legacy 调用形状是兼容性根节点

当前 handler 即使字段未出现，也会传 `experiment_ids=None`。虽然内置 abstract store 已更新签名，第三方 entry-point store 仍可能实现旧签名。新 keyword 会让所有旧 batch 请求失败。

建议：仅当 `has_experiment_ids` 为真时才传 `experiment_ids` keyword；否则保持原有 `batch_get_traces(trace_ids, None)` 和 `batch_get_trace_infos(trace_ids)` 调用形状。Copilot review body 中还有一个没有独立 ID 的 suppressed comment，指出 `batch_get_traces` 的相同问题，可随本组一并解决。

### C — scorer 的两个字段应维持互斥

当前实现与测试明确拒绝两个字段同时出现。这也符合 mprahl 在此前 thread 中提到的方向：`experiment_ids` 是通用的多 experiment 查询能力，不应通过 union 语义把 scope 放大。

建议：不恢复 merge 行为；更新 PR 描述中"`_list_scorers` merges …"为"两个字段互斥"。若 auth plugin 需要降级，应在 plugin 侧按 reviewer 早先建议设置其中一个字段。

### D — chunked trace 查询需先去重

`_query_trace_infos_in_batches()` 将每个 chunk 的结果连接起来。若同一 ID 进入不同 chunk，其 trace 会在每个 chunk 被查出一次；`batch_get_traces` 和 `batch_get_trace_infos` 都复用该 helper。

建议：在解析为整数后、分 chunk 前做保序去重；用小的 `_ID_CHUNK_SIZE` 测试重复 ID 跨 chunk，且同时断言 traces 与 trace infos 各只返回一次。

### E / I / J — 可以最后一起收尾

- E：PR appendix 更新成"Databricks-hosted backend 上，显式 `experiment_ids` 返回 400"，不再写成静默忽略。
- I：删除无关的 `batch_get_trace_infos` Databricks pre-check。
- J：把 "authorization scope" 改为中性的 "experiment ID filter / supplied IDs"。

### F / K — 不建议在本 PR 主动做的大重构

两条都在讨论 `list_active_experiment_ids` 的职责。当前 contract 是"验证**有界、调用方提供的** IDs"；而 `_search_active_experiment_ids` 是 workspace-aware 的开放式分页枚举。将 `None` 改为"列出全部 active"会扩张 abstract-store contract，并需要重新审视 workspace filtering 与非 SQL store 的实现。

建议：先回复说明维持现有职责边界，除非 mprahl 明确希望本 PR 接受该重构；不要把它混入上述 P0/P1 修复。

### G / H — 小改动，但先确认语义

- G：需要确认 malformed ID 的预期。若应"过滤掉"，则不能只删 handler validation；SQL 与默认 `AbstractStore` 的行为也要统一。若保留 400，应说明它是输入格式校验，并非 active-state 过滤。
- H：可低风险加 `ORDER BY SqlExperiment.experiment_id`，明确 `list_active_experiment_ids` 的确定性输出。

## 可批量处理的修改

1. `handlers.py` + `tests/server/test_handlers.py`：A 与 B，以及对应的 omitted/camelCase/legacy-store 用例。
2. `sqlalchemy_store.py` + trace store tests：D 和 J；顺便为 H 添加明确排序测试（若接受）。
3. PR 描述与 appendix：C、E。
4. I 可和 handlers 改动一起移除，但在代码行为确认后再做。

## 最后统一处理的 nit / style

- J 是纯措辞，最后处理即可。
- E 是文档与实现不一致，虽非代码 blocker，但应在提交前修正。
- H 是确定性/可维护性增强，不阻塞 A/B/D。
