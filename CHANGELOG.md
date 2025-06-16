# CHANGELOG

## 3.1 (2025-06-11)

MLflow 3 includes several major features and improvements

Features:

- [Tracking] MLflow 3.0 (#13211, @harupy)
- [Prompts] Add Custom Prompt Judges to `mlflow[databricks]` (#16097, @dbrx-euirim)
- [Artifacts / Model Registry / Tracking] Package model environment when registering model (#15783, @qyc)
- [Tracking] Add `MlflowSparkStudy` (#15418, @lu-wang-dl)
- [Scoring] Make `spark_udf` support DBConnect + DBR 15.4 / DBR dedicated cluster (#15968, @WeichenXu123)
- [Tracking] Lock model dependencies when logging a model using `uv` (#15875, @harupy)
- [Model Registry] Introduce `mlflow.genai.optimize_prompt` to optimize prompts (#15861, @TomeHirata)
- [Tracing] Support custom request/response preview (#15919, @B-Step62)
- [Tracking] Add integration for AutoGen > 0.4 (#14729, @TomeHirata)
- [Tracking] Support token tracking for OpenAI (#15870, @B-Step62)
- [Tracking] Support tracing `ResponsesAgent.predict_stream` (#15762, @bbqiu)
- [Tracking] Introduce client and fluent APIs for `LogLoggedModelParams` (#15717, @artjen)
- [Models] Support `predict_stream` in DSPy flavor (#15678, @TomeHirata)
- [Tracking] Record notebook and git metadata in trace metadata (#15650, @B-Step62)
- [Model Registry] Added `search_prompts` function to list all the prompts registered (#15445, @joelrobin18)
- [Models] Support compression for pyfunc log model (#14700, @antbbn)
- [Gateway] Add support for Gemini in AI Gateway (#15069, @joelrobin18)
- [Tracing] PydanticAI Autologging (#15553, @joelrobin18)
- [Tracking] Support setting databricks auth profile by `DATABRICKS_CONFIG_PROFILE` environment variable. (#15587, @WeichenXu123)
- [Tracking] create mlflow tracing for `smolagents` (#15574, @y-okt)
- [Artifacts / UI] Support for video artifacts (#15518, @joelrobin18)
- [Model Registry] Add `allow_missing` parameter in `load_prompt` (#15371, @joelrobin18)
- [Tracking] Emit a warning for `mlflow.get_artifact_uri()` usage outside active run (#12902, @Shashank1202)

Bug fixes:

- [GenAI] Add Databricks App resource (#15867, @aravind-segu)
- [Tracking] Support json-string for inputs/expectations column in Spark Dataframe (#16011, @B-Step62)
- [Tracking] Avoid generating traces from scorers during evaluation (#16004, @B-Step62)
- [GenAI] Allow multi inputs module in DSPy (#15859, @TomeHirata)
- [Tracking] Improve error handling if tracking URI is not set when running `mlflow gc` (#11773, @oleg-z)
- [Tracking] Trace search: Avoid spawning threads for span fetching if `include_spans=False` (#15634, @dbczumar)
- [Tracking] Fix `global_guideline_adherence` (#15572, @artjen)
- [Model Registry] Log `Resources` from `SystemAuthPolicy` in `CreateModelVersion` (#15485, @aravind-segu)
- [Models] `ResponsesAgent` interface update (#15601, #15741, @bbqiu)

Breaking changes:

- [Tracking] Move prompt registry APIs under `mlflow.genai.prompts` namespace (#16174, @B-Step62)
- [Model Registry] Default URI to databricks-uc when tracking URI is databricks & registry URI is unspecified (#16135, @dbczumar)
- [Tracking] Do not log SHAP explainer in `mlflow.evaluate` (#15827, @harupy)
- [Tracking] Update DataFrame schema returned from `mlflow.search_trace()` to be V3 format (#15643, @B-Step62)

Documentation updates:

- [Docs] Documentation revamp for MLflow 3.0 (#15954, @harupy)
- [Docs] Add Prompt Optimization Document Page (#15958, @TomeHirata)
- [Docs] Redesign API reference page (#15811, @besirovic)
- [Docs] MLflow 3 breaking changes list (#15716, @WeichenXu123)
- [Docs] Update Lighthouse signup and signin links (#15740, @BenWilson2)
- [Docs] Document models:/ URIs explicitly in OSS MLflow docs (#15727, @WeichenXu123)
- [Docs] Spark UDF Doc update (#15586, @WeichenXu123)

Small bug fixes and documentation updates:

#16193, #16192, #16171, #16119, #16036, #16130, #16081, #16101, #16047, #16086, #16077, #16045, #16065, #16067, #16063, #16061, #16058, #16050, #16043, #16034, #16033, #15966, #16025, #16015, #16002, #15970, #16001, #15999, #15942, #15960, #15955, #15951, #15939, #15885, #15883, #15890, #15887, #15874, #15869, #15846, #15845, #15826, #15834, #15822, #15830, #15796, #15821, #15818, #15817, #15805, #15804, #15798, #15793, #15797, #15782, #15775, #15772, #15790, #15773, #15776, #15756, #15767, #15766, #15765, #15746, #15747, #15748, #15751, #15743, #15731, #15720, #15722, #15670, #15614, #15715, #15677, #15708, #15673, #15680, #15686, #15671, #15657, #15669, #15664, #15675, #15667, #15666, #15668, #15651, #15649, #15647, #15640, #15638, #15630, #15627, #15624, #15622, #15558, #15610, #15577, #15575, #15545, #15576, #15559, #15563, #15555, #15557, #15548, #15551, #15547, #15542, #15536, #15524, #15531, #15525, #15520, #15521, #15502, #15499, #15442, #15426, #15315, #15392, #15397, #15399, #15394, #15358, #15352, #15349, #15328, #15336, #15335, @harupy; #16196, #16191, #16093, #16114, #16080, #16088, #16053, #15856, #16039, #15987, #16009, #16014, #16007, #15996, #15993, #15991, #15989, #15978, #15839, #15953, #15934, #15929, #15926, #15909, #15900, #15893, #15889, #15881, #15879, #15877, #15865, #15863, #15854, #15852, #15848, @copilot-swe-agent; #16178, #16153, #16155, #15823, #15754, #15794, #15800, #15799, #15615, #15777, #15726, #15752, #15745, #15753, #15738, #15681, #15684, #15682, #15702, #15679, #15623, #15645, #15612, #15533, #15607, #15522, @serena-ruan; #16177, #16167, #16168, #16166, #16152, #16144, #15920, #16134, #16128, #16098, #16059, #16024, #15974, #15917, #15676, #15750, @dbczumar; #16162, #16161, #16137, #16126, #16127, #16099, #16074, #16041, #16040, #16010, #15945, #15697, #15588, #15602, #15581, @rohitarun-db; #16150, #15984, #16125, #16102, #16062, #16060, #15986, #15985, #15983, #15982, #15980, #15763, @smoorjani; #16160, #16149, #16103, #15538, #16055, #16054, #16048, #16012, #16029, #16003, #15940, #15956, #15950, #15906, #15922, #15932, #15930, #15905, #15910, #15902, #15901, #15840, #15896, #15898, #15895, #15850, #15833, #15824, #15819, #15816, #15806, #15803, #15795, #15759, #15791, #15792, #15774, #15769, #15768, #15770, #15755, #15771, #15737, #15690, #15733, #15730, #15687, #15660, #15735, #15688, #15705, #15590, #15663, #15665, #15658, #15594, #15620, #15644, #15648, #15605, #15639, #15642, #15619, #15618, #15611, #15597, #15589, #15580, #15593, #15437, #15584, #15582, #15448, #15351, #15317, #15353, #15320, #15319, @B-Step62; #16151, #16142, #16111, #16106, #16051, #16046, #16044, #15971, #15957, #15810, #15749, #15706, #15683, #15728, #15732, #15707, #15621, #15567, #15566, #15523, #15479, #15404, #15400, #15378, @TomeHirata; #16026, #16072, @AveshCSingh; #15967, @euirim; #15884, #15924, #15395, #15393, #15390, @daniellok-db; #15786, @rahuja23; #15734, @lhrotk; #15809, #15739, #15695, #15654, #15694, #15655, #15653, #15608, #15543, #15573, @dhruyads; #15596, @mrharishkumar; #15742, #15723, #15633, #15606, @ShaylanDias; #15703, #15637, #15613, #15473, @joelrobin18; #15636, #15659, #15616, #15617, @raymondzhou-db; #15674, #15598, #15357, #15586, @WeichenXu123; #15691, @artjen; #15698, @prithvikannan; #15631, @hubertzub-db; #15569, @Anand1923; #15578, @y-okt; #14790, @singh-kristian; #14129, @jamblejoe; #15552, @BenWilson2; #14197, @clarachristiansen; #15505, @Conor0Callaghan; #15509, @tr33k; #15507, @vzamboulingame; #15459, @UnMelow; #13991, @abhishekpawar1060; #12161, @zhouyou9505; #15293, @tornikeo

## < 3.0

See [MLflow 2.x changelog](changelogs/CHANGELOG_2.x.md) for changes in MLflow < 3.0.
