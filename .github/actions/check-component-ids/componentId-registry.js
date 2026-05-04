/**
 * Curated registry of all componentIds used in the MLflow UI.
 *
 * Every static componentId string literal in non-test source files must
 * have an entry here. The CI job `check-component-ids` verifies this
 * bidirectionally: code IDs must be in the registry, and registry
 * entries must exist in code.
 *
 * Format: key = componentId string, value = optional description of the
 * component (blank by default, especially for generated entries)
 */
module.exports = {
  // -- Codegen (auto-generated) --
  "codegen_mlflow_app_src_common_components_darkthemeswitch.tsx_32": "",
  "codegen_mlflow_app_src_common_components_editablenote.tsx_114": "",
  "codegen_mlflow_app_src_common_components_editablenote.tsx_124": "",
  "codegen_mlflow_app_src_common_components_editablenote.tsx_178": "",
  "codegen_mlflow_app_src_common_components_editabletagstableview.tsx_107": "",
  "codegen_mlflow_app_src_common_components_editabletagstableview.tsx_117": "",
  "codegen_mlflow_app_src_common_components_editabletagstableview.tsx_127": "",
  "codegen_mlflow_app_src_common_components_iconbutton.tsx_20": "",
  "codegen_mlflow_app_src_common_components_keyvaluetag.tsx_60": "",
  "codegen_mlflow_app_src_common_components_keyvaluetagfullviewmodal.tsx_17": "",
  "codegen_mlflow_app_src_common_components_keyvaluetagseditorcell.tsx_29": "",
  "codegen_mlflow_app_src_common_components_keyvaluetagseditorcell.tsx_37": "",
  "codegen_mlflow_app_src_common_components_previewsidebar.tsx_67": "",
  "codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_120": "",
  "codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_131": "",
  "codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_145": "",
  "codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_151": "",
  "codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_228": "",
  "codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_50": "",
  "codegen_mlflow_app_src_common_components_trimmedtext.tsx_30": "",
  "codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_135": "",
  "codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_147": "",
  "codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_174": "",
  "codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_223": "",
  "codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_248": "",
  "codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_306": "",
  "codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_309": "",
  "codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_316": "",
  "codegen_mlflow_app_src_common_hooks_useeditkeyvaluetagsmodal.tsx_324": "",
  "codegen_mlflow_app_src_experiment-tracking_components_artifact-view-components_showartifactloggedtableview.tsx_181":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_artifact-view-components_showartifactloggedtableview.tsx_223":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_artifact-view-components_showartifactloggedtableview.tsx_315":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_artifact-view-components_showartifactloggedtableview.tsx_331":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_artifactview.tsx_288": "",
  "codegen_mlflow_app_src_experiment-tracking_components_artifactview.tsx_337": "",
  "codegen_mlflow_app_src_experiment-tracking_components_comparerunbox.tsx_46": "",
  "codegen_mlflow_app_src_experiment-tracking_components_compareruncontour.tsx_282": "",
  "codegen_mlflow_app_src_experiment-tracking_components_compareruncontour.tsx_299": "",
  "codegen_mlflow_app_src_experiment-tracking_components_comparerunscatter.tsx_182": "",
  "codegen_mlflow_app_src_experiment-tracking_components_comparerunview.tsx_570": "",
  "codegen_mlflow_app_src_experiment-tracking_components_comparerunview.tsx_581": "",
  "codegen_mlflow_app_src_experiment-tracking_components_comparerunview.tsx_592": "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationcellevaluatebutton.tsx_59":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationcreatepromptrunoutput.tsx_144":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationcreatepromptrunoutput.tsx_85":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationcreatepromptrunoutput.tsx_99":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadercellrenderer.tsx_112":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadercellrenderer.tsx_118":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadercellrenderer.tsx_143":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadercellrenderer.tsx_150":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheaderdatasetindicator.tsx_37":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheaderdatasetindicator.tsx_49":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheaderdatasetindicator.tsx_51":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheaderdatasetindicator.tsx_66":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadermodelindicator.tsx_107":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadermodelindicator.tsx_115":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationtableactionscellrenderer.tsx_37":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationtableactionscolumnrenderer.tsx_22":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_createnotebookrunmodal.tsx_111":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_createnotebookrunmodal.tsx_117":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationartifactcompareview.tsx_358":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationartifactcompareview.tsx_414":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationartifactcompareview.tsx_433":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationartifactcompareview.tsx_465":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationartifactviewemptystate.tsx_48":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptparameters.tsx_107":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptparameters.tsx_28":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptparameters.tsx_39":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_541":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_589":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_596":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_597":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_638":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_678":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_694":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_695":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_736":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodalexamples.tsx_42":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodalexamples.tsx_48":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodalexamples.tsx_90":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_hooks_useevaluationaddnewinputsmodal.tsx_57":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_hooks_useevaluationaddnewinputsmodal.tsx_99":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_hooks_useevaluationartifactwriteback.tsx_102":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_hooks_useevaluationartifactwriteback.tsx_110":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluations_components_evaluationsreviewassessmentssection.tsx_149":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluations_components_evaluationsreviewassessmentupsertform.tsx_124":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluations_components_evaluationsreviewassessmentupsertform.tsx_160":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluations_components_evaluationsreviewretrievalsection.tsx_30":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluations_components_evaluationsreviewretrievalsection.tsx_32":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_evaluations_evaluationsoverview.tsx_576":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_experimentviewdescriptionnotes.tsx_114":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_experimentviewdescriptionnotes.tsx_120":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_experimentviewdescriptionnotes.tsx_126":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_experimentviewdescriptionnotes.tsx_141":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_experimentviewnotes.tsx_57":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentgetsharelinkmodal.tsx_101":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentgetsharelinkmodal.tsx_115":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentviewheadersharebutton.tsx_44":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_datasetscellrenderer.tsx_172":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_datasetscellrenderer.tsx_184":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_datasetscellrenderer.tsx_49":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_datasetscellrenderer.tsx_56":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_datasetscellrenderer.tsx_75":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_groupparentcellrenderer.tsx_109":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_groupparentcellrenderer.tsx_136":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_loadmorerowrenderer.tsx_20":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_modelscellrenderer.tsx_49":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_rowactionsheadercellrenderer.tsx_52":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_runnamecellrenderer.tsx_46":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetdrawer.tsx_206":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetdrawer.tsx_81":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetlink.tsx_19_1":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetlink.tsx_19_2":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetschema.tsx_92":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetschematable.tsx_57":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetschematable.tsx_58":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetsourceurl.tsx_34":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewdatasetwithcontext.tsx_41":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscolumnselector.tsx_300":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscolumnselector.tsx_315":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrols.tsx_175":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactions.tsx_110":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactions.tsx_117":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactions.tsx_126":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactions.tsx_136":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactionsaddnewtagmodal.tsx_34":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactionsaddnewtagmodal.tsx_51":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactionsaddnewtagmodal.tsx_78":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactionsselecttags.tsx_162":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_184":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_201":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_211":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_217":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_248":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_289":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_329":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_338":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_362":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_382":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_402":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_403":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_415":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_461":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_469":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_time_button":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsemptytable.tsx_35":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_168":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_191":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_233":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_244":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_280":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_302":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_306":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_314":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_330":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_342":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_349":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_426":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_436":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunssortselectorv2.tsx_137":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunssortselectorv2.tsx_151":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunssortselectorv2.tsx_97":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunstableaddcolumncta.tsx_218":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_runssearchautocomplete.tsx_212":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_runssearchautocomplete.tsx_236":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_runssearchautocomplete.tsx_310":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_metricchartsaccordion.tsx_82": "",
  "codegen_mlflow_app_src_experiment-tracking_components_metricsplotcontrols.tsx_120": "",
  "codegen_mlflow_app_src_experiment-tracking_components_metricsplotcontrols.tsx_154": "",
  "codegen_mlflow_app_src_experiment-tracking_components_metricsplotcontrols.tsx_220": "",
  "codegen_mlflow_app_src_experiment-tracking_components_metricsplotcontrols.tsx_222": "",
  "codegen_mlflow_app_src_experiment-tracking_components_modals_createexperimentform.tsx_51": "",
  "codegen_mlflow_app_src_experiment-tracking_components_modals_createexperimentform.tsx_71": "",
  "codegen_mlflow_app_src_experiment-tracking_components_modals_getlinkmodal.tsx_21": "",
  "codegen_mlflow_app_src_experiment-tracking_components_modals_renameform.tsx_69": "",
  "codegen_mlflow_app_src_experiment-tracking_components_parallelcoordinatesplotcontrols.tsx_84":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewdatasetbox.tsx_16":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewdatasetbox.tsx_70":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewdatasetbox.tsx_81":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewdescriptionbox.tsx_46":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewmetricstable.tsx_186":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewmetricstable.tsx_312":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewparamstable.tsx_213":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewparamstable.tsx_244":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewparamstable.tsx_74":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewregisteredmodelsbox.tsx_40":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewsourcebox.tsx_48":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewstatusbox.tsx_81":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_195":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_231":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_50":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_58":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_80":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_89":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_90":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewmetricchartsv2.tsx_244":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_262":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_288":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_291":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_298":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_316":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_324":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_334":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_340":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_344":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_350":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_runschartsparallelchartcard.tsx_293":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_runschartsparallelchartcard.tsx_300":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_imagegridmultiplekeyplot.tsx_44":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_imagegridmultiplekeyplot.tsx_52":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfiguredifferencechart.tsx_129":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfiguredifferencechart.tsx_138":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfiguredifferencechart.tsx_157":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfiguredifferencechart.tsx_98":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigureimagechart.tsx_84":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_436":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_474":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_494":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_524":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_628":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_682":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_703":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_716":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_747":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_838":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_112":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_126":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_42":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_56":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_70":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_84":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_98":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsconfiguremodal.tsx_232":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsconfiguremodal.tsx_296":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsfilterinput.tsx_30":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsfullscreenmodal.tsx_53":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsglobalchartsettingsdropdown.tsx_118":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsglobalchartsettingsdropdown.tsx_44":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsglobalchartsettingsdropdown.tsx_68":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsglobalchartsettingsdropdown.tsx_78":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsglobalchartsettingsdropdown.tsx_88":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsnodatafoundindicator.tsx_31":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsyaxismetricandexpressionselector.tsx_122":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsyaxismetricandexpressionselector.tsx_221":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_sections_runschartssectionheader.tsx_220":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_sections_runschartssectionheader.tsx_321":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_sections_runschartssectionheader.tsx_327":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_sections_runschartssectionheader.tsx_333":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_sections_runschartssectionheader.tsx_351":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-charts_hooks_userunschartstooltip.stories.tsx_42":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-compare_cards_chartcard.common.tsx_158":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-compare_runscompareaddchartmenu.tsx_19":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-compare_runscomparetooltipbody.tsx_259":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-compare_runscomparetooltipbody.tsx_282":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-compare_runscomparetooltipbody.tsx_302":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-compare_sections_runscomparesectionaccordion.tsx_405":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-compare_sections_runscomparesectionheader.tsx_246":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-compare_sections_runscomparesectionheader.tsx_251":
    "",
  "codegen_mlflow_app_src_experiment-tracking_components_runs-compare_sections_runscomparesectionheader.tsx_288":
    "",
  "codegen_mlflow_app_src_model-registry_components_CreateModelButton.tsx_28": "",
  "codegen_mlflow_app_src_model-registry_components_aliases_modelstablealiasedversionscell.tsx_47":
    "",
  "codegen_mlflow_app_src_model-registry_components_aliases_modelstablealiasedversionscell.tsx_57":
    "",
  "codegen_mlflow_app_src_model-registry_components_aliases_modelversionaliastag.tsx_23": "",
  "codegen_mlflow_app_src_model-registry_components_aliases_modelversiontablealiasescell.tsx_30":
    "",
  "codegen_mlflow_app_src_model-registry_components_aliases_modelversiontablealiasescell.tsx_41":
    "",
  "codegen_mlflow_app_src_model-registry_components_aliases_modelversionviewaliaseditor.tsx_29": "",
  "codegen_mlflow_app_src_model-registry_components_aliases_modelversionviewaliaseditor.tsx_37": "",
  "codegen_mlflow_app_src_model-registry_components_createmodelform.tsx_62": "",
  "codegen_mlflow_app_src_model-registry_components_model-list_modellistfilters.tsx_118": "",
  "codegen_mlflow_app_src_model-registry_components_model-list_modellistfilters.tsx_152": "",
  "codegen_mlflow_app_src_model-registry_components_model-list_modellistfilters.tsx_46": "",
  "codegen_mlflow_app_src_model-registry_components_model-list_modellistfilters.tsx_61": "",
  "codegen_mlflow_app_src_model-registry_components_model-list_modellisttable.tsx_412": "",
  "codegen_mlflow_app_src_model-registry_components_model-list_modellisttable.tsx_learn_more": "",
  "codegen_mlflow_app_src_model-registry_components_model-list_modeltablecellrenderers.tsx_65": "",
  "codegen_mlflow_app_src_model-registry_components_modellistview.tsx_305": "",
  "codegen_mlflow_app_src_model-registry_components_modelsnextuipromomodal.tsx_15": "",
  "codegen_mlflow_app_src_model-registry_components_modelsnextuipromomodal.tsx_26": "",
  "codegen_mlflow_app_src_model-registry_components_modelsnextuipromomodal.tsx_32": "",
  "codegen_mlflow_app_src_model-registry_components_modelsnextuitoggleswitch.tsx_39": "",
  "codegen_mlflow_app_src_model-registry_components_modelsnextuitoggleswitch.tsx_50": "",
  "codegen_mlflow_app_src_model-registry_components_modelsnextuitoggleswitch.tsx_74": "",
  "codegen_mlflow_app_src_model-registry_components_modelversiontable.tsx_425": "",
  "codegen_mlflow_app_src_model-registry_components_modelversiontable.tsx_450": "",
  "codegen_mlflow_app_src_model-registry_components_modelversiontable.tsx_458": "",
  "codegen_mlflow_app_src_model-registry_components_modelversiontable.tsx_477": "",
  "codegen_mlflow_app_src_model-registry_components_modelversionview.tsx_301": "",
  "codegen_mlflow_app_src_model-registry_components_modelversionview.tsx_516": "",
  "codegen_mlflow_app_src_model-registry_components_modelversionview_tsx_394": "",
  "codegen_mlflow_app_src_model-registry_components_modelview.tsx_467": "",
  "codegen_mlflow_app_src_model-registry_components_modelview.tsx_600": "",
  "codegen_mlflow_app_src_model-registry_components_modelview.tsx_619": "",
  "codegen_mlflow_app_src_model-registry_components_modelview.tsx_646": "",
  "codegen_mlflow_app_src_model-registry_components_modelview.tsx_662": "",
  "codegen_mlflow_app_src_model-registry_components_promotemodelbutton.tsx_140": "",
  "codegen_mlflow_app_src_model-registry_components_promotemodelbutton.tsx_165": "",
  "codegen_mlflow_app_src_model-registry_components_registermodel.tsx_242": "",
  "codegen_mlflow_app_src_model-registry_components_registermodel.tsx_248": "",
  "codegen_mlflow_app_src_model-registry_components_registermodel.tsx_261": "",
  "codegen_mlflow_app_src_model-registry_components_registermodelform.tsx_132": "",
  "codegen_mlflow_app_src_model-registry_constants.tsx_37": "",
  "codegen_mlflow_app_src_model-registry_constants.tsx_38": "",
  "codegen_mlflow_app_src_model-registry_constants.tsx_39": "",
  "codegen_mlflow_app_src_model-registry_constants.tsx_40": "",
  "codegen_mlflow_app_src_shared_building_blocks_copybox.tsx_18": "",
  "codegen_mlflow_app_src_shared_building_blocks_pageheader.tsx_54": "",
  "codegen_mlflow_app_src_shared_building_blocks_previewbadge.tsx_14": "",
  codegen_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_cancel:
    "",
  codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_hooks_useunifiedtracetagsmodal_121:
    "",
  codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_hooks_useunifiedtracetagsmodal_130:
    "",
  codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_hooks_useunifiedtracetagsmodal_141:
    "",
  codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_hooks_useunifiedtracetagsmodal_157:
    "",
  codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_hooks_useunifiedtracetagsmodal_207:
    "",
  codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_timeline_tree_timelinetreefilterbutton_111:
    "",
  codegen_no_dynamic_js_packages_web_shared_src_model_trace_explorer_timeline_tree_timelinetreefilterbutton_83:
    "",
  codegen_no_dynamic_mlflow_web_js_src_common_hooks_usetagassignmentmodal_115: "",
  codegen_no_dynamic_mlflow_web_js_src_common_hooks_usetagassignmentmodal_82: "",
  codegen_no_dynamic_mlflow_web_js_src_common_hooks_usetagassignmentmodal_91: "",
  codegen_no_dynamic_mlflow_web_js_src_common_hooks_usetagassignmentmodal_99: "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_evaluations_evaluationruncompareselector_112:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_evaluations_evaluationruncompareselector_190:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_experiment_page_components_experimentlistviewtagsfilter_69:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_experiment_page_components_experimentlistviewtagsfilter_87:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_experiment_page_components_experimentlistviewtagsfilter_96:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_experiment_page_components_header_experimentviewheaderkindselector_113:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_traces_quickstart_tracesviewtablenotracesquickstart_46:
    "",
  "codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_traces_quickstart_tracetablequickstart.utils_366":
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_evaluation_runs_experimentevaluationrunstablecellrenderers_284:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_page_tabs_side_nav_experimentpagesidenavsection_93:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_customcodescorerformrenderer_152:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_customcodescorerformrenderer_209:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_deletescorermodalrenderer_28:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_deletescorermodalrenderer_46:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_178:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_224:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_234:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_263:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_271:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_316:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_52:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorercardrenderer_106:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorercardrenderer_123:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorercardrenderer_179:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorercardrenderer_41:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorercardrenderer_45:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorercardrenderer_85:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scoreremptystaterenderer_59:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorerformrenderer_140:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorerformrenderer_293:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorerformrenderer_298:
    "",
  codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorermodalrenderer_29:
    "",
  "codegen_web-shared_src_copy_copyactionbutton.tsx_17": "",
  "codegen_web-shared_src_snippet_actions_snippetactionbutton.tsx_26": "",
  "codegen_web-shared_src_snippet_actions_snippetactionbutton.tsx_33": "",
  "codegen_webapp_js_genai_util_markdown.tsx_71": "",

  // -- Other --
  "TagAssignmentKey.Default.Input": "",
  "TagAssignmentValue.Default.Input": "",
  account: "",
  "account.change_password_button": "",
  "account.change_password_modal": "",
  "account.change_password_modal.error": "",
  "account.confirm_password": "",
  "account.current_password": "",
  "account.error": "",
  "account.new_password": "",
  "account.no_user": "",
  "account.role_admin_tag": "",
  "account.roles.admin_header": "",
  "account.roles.name_header": "",
  "account.roles.workspace_header": "",
  "account.roles_error": "",
  "account.tabs": "",
  cancel: "",
  "categorical-aggregate-chart-more-items": "",
  "databricks-experiment-tracking-prompt-edit-tags-button": "",
  "delete-run-modal": "",
  "delete-selected": "",
  "delete-selected-children": "",
  "discovery.data_explorer.entity_comment.show_comment_text_toggle": "",
  "endpoint-tags-section.remove-button": "",
  "eval-tab.delete_traces-modal": "",
  "experiment-evaluation-monitoring-end-date-picker": "",
  "experiment-evaluation-monitoring-start-date-picker": "",
  fullscreen_button_chartcard: "",
  "genai.util.markdown-copy-code-block": "",
  "graph-view-span-navigator-next": "",
  "graph-view-span-navigator-prev": "",
  "graph-view-toolbar.expand": "",
  "graph-view-toolbar.expand-button": "",
  "graph-view-toolbar.fit-view": "",
  "graph-view-toolbar.fit-view-button": "",
  "graph-view-toolbar.zoom-in": "",
  "graph-view-toolbar.zoom-in-button": "",
  "graph-view-toolbar.zoom-out": "",
  "graph-view-toolbar.zoom-out-button": "",
  "mlflow_header.toggle_sidebar_button": "",
  "open-modal": "",
  promptType: "",
  "storybook.long-form.description": "",
  "storybook.long-form.model": "",
  "storybook.long-form.name": "",
  "storybook.long-form.provider": "",
  "traces-v3-empty-state-button": "",
  "virtualized-table-header": "",
  "web-shared.genai-traces-table.evaluations-review-assessment.tooltip": "",
  "web-shared.genai-traces-table.key-value-tag.full-view-tooltip": "",
  "web-shared.time-ago": "",
  workspace_selector: "",
  "workspace_selector.tooltip": "",

  // -- mlflow.artifact_view --
  "mlflow.artifact_view.download_artifact": "",
  "mlflow.artifact_view.markdown_render_mode": "",
  "mlflow.artifact_view.markdown_rendered_tooltip": "",
  "mlflow.artifact_view.markdown_source_tooltip": "",

  // -- mlflow.artifacts --
  "mlflow.artifacts.logged_model_fallback_info": "",
  "mlflow.artifacts.model_version.link": "",
  "mlflow.artifacts.model_version.status": "",

  // -- mlflow.assistant --
  "mlflow.assistant.chat_panel.beta": "",
  "mlflow.assistant.chat_panel.close": "",
  "mlflow.assistant.chat_panel.close.tooltip": "",
  "mlflow.assistant.chat_panel.context.dataset": "",
  "mlflow.assistant.chat_panel.context.model": "",
  "mlflow.assistant.chat_panel.context.prompt": "",
  "mlflow.assistant.chat_panel.context.run": "",
  "mlflow.assistant.chat_panel.context.scorer": "",
  "mlflow.assistant.chat_panel.context.session": "",
  "mlflow.assistant.chat_panel.context.trace": "",
  "mlflow.assistant.chat_panel.copy": "",
  "mlflow.assistant.chat_panel.copy.tooltip": "",
  "mlflow.assistant.chat_panel.regenerate": "",
  "mlflow.assistant.chat_panel.regenerate.tooltip": "",
  "mlflow.assistant.chat_panel.remote_close": "",
  "mlflow.assistant.chat_panel.reset": "",
  "mlflow.assistant.chat_panel.reset.tooltip": "",
  "mlflow.assistant.chat_panel.send": "",
  "mlflow.assistant.chat_panel.settings": "",
  "mlflow.assistant.chat_panel.settings.tooltip": "",
  "mlflow.assistant.chat_panel.setup": "",
  "mlflow.assistant.chat_panel.suggestion.card": "",
  "mlflow.assistant.icon_button": "",
  "mlflow.assistant.icon_button.tooltip": "",
  "mlflow.assistant.setup.complete.start_chatting": "",
  "mlflow.assistant.setup.connection.back": "",
  "mlflow.assistant.setup.connection.check_again": "",
  "mlflow.assistant.setup.connection.continue": "",
  "mlflow.assistant.setup.connection.copy": "",
  "mlflow.assistant.setup.connection.learn_more": "",
  "mlflow.assistant.setup.footer.back": "",
  "mlflow.assistant.setup.footer.next": "",
  "mlflow.assistant.setup.project.custom_skills_path": "",
  "mlflow.assistant.setup.project.error": "",
  "mlflow.assistant.setup.project.path_input": "",
  "mlflow.assistant.setup.project.perm_edit_files": "",
  "mlflow.assistant.setup.project.perm_edit_files_tooltip": "",
  "mlflow.assistant.setup.project.perm_full": "",
  "mlflow.assistant.setup.project.perm_full_tooltip": "",
  "mlflow.assistant.setup.project.perm_mlflow_cli": "",
  "mlflow.assistant.setup.project.perm_mlflow_cli_tooltip": "",
  "mlflow.assistant.setup.project.perm_read_docs": "",
  "mlflow.assistant.setup.project.perm_read_docs_tooltip": "",
  "mlflow.assistant.setup.project.skills_custom": "",
  "mlflow.assistant.setup.project.skills_global": "",
  "mlflow.assistant.setup.project.skills_link": "",
  "mlflow.assistant.setup.project.skills_location": "",
  "mlflow.assistant.setup.project.skills_project": "",
  "mlflow.assistant.setup.provider.continue": "",

  // -- mlflow.charts --
  "mlflow.charts.bar_card_title.dataset_tag": "",
  "mlflow.charts.chart_configure.metric_with_dataset_select": "",
  "mlflow.charts.chart_configure.metric_with_dataset_select.tag": "",
  "mlflow.charts.controls.global_chart_setup_dropdown": "",
  "mlflow.charts.difference_chart_configure_button": "",
  "mlflow.charts.difference_plot.expand_button": "",
  "mlflow.charts.difference_plot.header": "",
  "mlflow.charts.difference_plot.overflow_menu.set_as_baseline": "",
  "mlflow.charts.difference_plot.overflow_menu.trigger": "",
  "mlflow.charts.image-plot.run-name-tooltip": "",
  "mlflow.charts.line-chart-expressions-add-new": "",
  "mlflow.charts.line-chart-expressions-remove": "",
  "mlflow.charts.line_chart_configure.display_points.auto.tooltip": "",
  "mlflow.charts.line_chart_configure.x_axis_max": "",
  "mlflow.charts.line_chart_configure.x_axis_min": "",
  "mlflow.charts.line_chart_configure.x_axis_type.relative_time.tooltip": "",
  "mlflow.charts.line_chart_configure.x_axis_type.wall_time.tooltip": "",
  "mlflow.charts.line_chart_configure.y_axis_max": "",
  "mlflow.charts.line_chart_configure.y_axis_min": "",
  "mlflow.charts.parallel_coords_chart_configure_button": "",
  "mlflow.charts.scatter_card_title.dataset_tag": "",
  "mlflow.charts.tool_error_rate": "",
  "mlflow.charts.tool_error_rate.tool_selector": "",
  "mlflow.charts.tool_error_rate_section": "",
  "mlflow.charts.tool_latency": "",
  "mlflow.charts.tool_latency.tool_selector": "",
  "mlflow.charts.tool_performance_summary": "",
  "mlflow.charts.tool_usage": "",
  "mlflow.charts.tool_usage.tool_selector": "",
  "mlflow.charts.trace_cost_breakdown": "",
  "mlflow.charts.trace_cost_breakdown.dimension": "",
  "mlflow.charts.trace_errors": "",
  "mlflow.charts.trace_latency": "",
  "mlflow.charts.trace_requests": "",
  "mlflow.charts.trace_requests.zoom_out": "",
  "mlflow.charts.trace_token_stats": "",
  "mlflow.charts.trace_token_usage": "",

  // -- mlflow.chat-sessions --
  "mlflow.chat-sessions.actions-dropdown": "",
  "mlflow.chat-sessions.actions-dropdown-tooltip": "",
  "mlflow.chat-sessions.copy-session-id": "",
  "mlflow.chat-sessions.delete-sessions": "",
  "mlflow.chat-sessions.session-header-label": "",
  "mlflow.chat-sessions.session-id-tag": "",
  "mlflow.chat-sessions.table-column-selector": "",
  "mlflow.chat-sessions.table-header": "",
  "mlflow.chat-sessions.table-header-checkbox": "",
  "mlflow.chat-sessions.table-row-checkbox": "",

  // -- mlflow.chat_sessions --
  "mlflow.chat_sessions.empty_state.example_code_copy": "",
  "mlflow.chat_sessions.empty_state.learn_more_link": "",

  // -- mlflow.common --
  "mlflow.common.components.editable-note.tooltip-icon": "",
  "mlflow.common.components.key-value-tag.tooltip": "",
  "mlflow.common.components.tag-select-dropdown.add-new-tag-tooltip": "",
  "mlflow.common.error_view.fallback_link": "",
  "mlflow.common.expandable_cell": "",
  "mlflow.common.hooks.useeditkeyvaluetagsmodal.add-tag-tooltip": "",
  "mlflow.common.hooks.useeditkeyvaluetagsmodal.tooltip": "",

  // -- mlflow.compare-model-versions --
  "mlflow.compare-model-versions.plots-tabs": "",

  // -- mlflow.compare-runs --
  "mlflow.compare-runs.visualizations-tabs": "",

  // -- mlflow.compare_runs --
  "mlflow.compare_runs.data_cell": "",
  "mlflow.compare_runs.metric_table.cell": "",

  // -- mlflow.create-evaluation-dataset-modal --
  "mlflow.create-evaluation-dataset-modal": "",
  "mlflow.create-evaluation-dataset-modal.dataset-name": "",

  // -- mlflow.create-notebook-run-modal --
  "mlflow.create-notebook-run-modal.tabs": "",

  // -- mlflow.dataset_drawer --
  "mlflow.dataset_drawer.dataset_name_tooltip": "",

  // -- mlflow.detect_issues --
  "mlflow.detect_issues.guidance": "",
  "mlflow.detect_issues.guidance.dismiss": "",
  "mlflow.detect_issues.guidance.got_it": "",

  // -- mlflow.edit-aliases-modal --
  "mlflow.edit-aliases-modal": "",
  "mlflow.edit-aliases-modal.cancel-button": "",
  "mlflow.edit-aliases-modal.conflicted-alias-alert": "",
  "mlflow.edit-aliases-modal.error-alert": "",
  "mlflow.edit-aliases-modal.exceeding-limit-alert": "",
  "mlflow.edit-aliases-modal.save-button": "",

  // -- mlflow.endpoint-selector --
  "mlflow.endpoint-selector.deleted-endpoint-tooltip": "",
  "mlflow.endpoint-selector.endpoints-error": "",
  "mlflow.endpoint-selector.select": "",

  // -- mlflow.eval-dataset-records --
  "mlflow.eval-dataset-records.column-header": "",

  // -- mlflow.eval-datasets --
  "mlflow.eval-datasets.column-header": "",
  "mlflow.eval-datasets.create-dataset-button": "",
  "mlflow.eval-datasets.dataset-actions-menu": "",
  "mlflow.eval-datasets.dataset-id": "",
  "mlflow.eval-datasets.dataset-id-tooltip": "",
  "mlflow.eval-datasets.dataset-name-cell": "",
  "mlflow.eval-datasets.delete-dataset-menu-option": "",
  "mlflow.eval-datasets.last-updated-cell-tooltip": "",
  "mlflow.eval-datasets.learn-more-link": "",
  "mlflow.eval-datasets.records-toolbar.column-checkbox": "",
  "mlflow.eval-datasets.records-toolbar.columns-toggle": "",
  "mlflow.eval-datasets.records-toolbar.row-size-radio": "",
  "mlflow.eval-datasets.records-toolbar.row-size-toggle": "",
  "mlflow.eval-datasets.records-toolbar.search-input": "",
  "mlflow.eval-datasets.search-input": "",
  "mlflow.eval-datasets.table-column-selector-button": "",
  "mlflow.eval-datasets.table-column-selector-checkbox": "",
  "mlflow.eval-datasets.table-refresh-button": "",

  // -- mlflow.eval-runs --
  "mlflow.eval-runs.actions-button": "",
  "mlflow.eval-runs.actions.compare": "",
  "mlflow.eval-runs.actions.delete": "",
  "mlflow.eval-runs.charts-mode-toggle-tooltip": "",
  "mlflow.eval-runs.checkbox-cell": "",
  "mlflow.eval-runs.compare-button": "",
  "mlflow.eval-runs.compare-button.tooltip": "",
  "mlflow.eval-runs.dataset-cell": "",
  "mlflow.eval-runs.dataset-cell-tooltip": "",
  "mlflow.eval-runs.empty-state.learn-more-link": "",
  "mlflow.eval-runs.group-expand-button": "",
  "mlflow.eval-runs.group-tag": "",
  "mlflow.eval-runs.header": "",
  "mlflow.eval-runs.issue-detection-run-icon-tooltip": "",
  "mlflow.eval-runs.model-version-cell": "",
  "mlflow.eval-runs.model-version-cell-tooltip": "",
  "mlflow.eval-runs.page-mode-selector": "",
  "mlflow.eval-runs.run-name-cell": "",
  "mlflow.eval-runs.run-name-cell.open-run-page": "",
  "mlflow.eval-runs.run-name-cell.tooltip": "",
  "mlflow.eval-runs.runs-delete-modal": "",
  "mlflow.eval-runs.start-run-button": "",
  "mlflow.eval-runs.start-run-modal": "",
  "mlflow.eval-runs.table-column-selector": "",
  "mlflow.eval-runs.table-refresh-button": "",
  "mlflow.eval-runs.table-refresh-button.tooltip": "",
  "mlflow.eval-runs.traces-mode-toggle-tooltip": "",
  "mlflow.eval-runs.visibility-mode-selector": "",

  // -- mlflow.evaluations_overview --
  "mlflow.evaluations_overview.column_selector_dropdown": "",

  // -- mlflow.evaluations_overview_grouped --
  "mlflow.evaluations_overview_grouped.column_selector_dropdown": "",

  // -- mlflow.evaluations_review --
  "mlflow.evaluations_review.cancel_edited_assessment_button": "",
  "mlflow.evaluations_review.cancel_override_assessments_button": "",
  "mlflow.evaluations_review.column_count": "",
  "mlflow.evaluations_review.confirm_edited_assessment_button": "",
  "mlflow.evaluations_review.discard_pending_assessments_button": "",
  "mlflow.evaluations_review.edit_assessment_button": "",
  "mlflow.evaluations_review.evaluation_error_alert": "",
  "mlflow.evaluations_review.mark_as_reviewed_button": "",
  "mlflow.evaluations_review.modal": "",
  "mlflow.evaluations_review.modal.add_to_dataset": "",
  "mlflow.evaluations_review.modal.add_to_evaluation_dataset": "",
  "mlflow.evaluations_review.modal.next_eval": "",
  "mlflow.evaluations_review.modal.previous_eval": "",
  "mlflow.evaluations_review.modal.share-button": "",
  "mlflow.evaluations_review.modal.share-notification": "",
  "mlflow.evaluations_review.modal.share-tooltip": "",
  "mlflow.evaluations_review.next_evaluation_result_button": "",
  "mlflow.evaluations_review.rca_pill": "",
  "mlflow.evaluations_review.reopen_review_button": "",
  "mlflow.evaluations_review.save_pending_assessments_button": "",
  "mlflow.evaluations_review.see_detailed_trace_view_button": "",
  "mlflow.evaluations_review.see_detailed_trace_view_tooltip": "",
  "mlflow.evaluations_review.table_ui.add_filter_button": "",
  "mlflow.evaluations_review.table_ui.apply_filters_button": "",
  "mlflow.evaluations_review.table_ui.compare_to_run_button": "",
  "mlflow.evaluations_review.table_ui.evaluation_id_link": "",
  "mlflow.evaluations_review.table_ui.filter_button": "",
  "mlflow.evaluations_review.table_ui.filter_column": "",
  "mlflow.evaluations_review.table_ui.filter_control": "",
  "mlflow.evaluations_review.table_ui.filter_delete_button": "",
  "mlflow.evaluations_review.table_ui.filter_input": "",
  "mlflow.evaluations_review.table_ui.filter_key": "",
  "mlflow.evaluations_review.table_ui.filter_operator": "",
  "mlflow.evaluations_review.table_ui.filter_value": "",
  "mlflow.evaluations_review.table_ui.filter_value_numeric": "",
  "mlflow.evaluations_review.textbox.copy": "",
  "mlflow.evaluations_review.trace_data_drawer": "",

  // -- mlflow.experiment --
  "mlflow.experiment.chat-session.metrics.goal-tag": "",
  "mlflow.experiment.chat-session.metrics.goal-tooltip": "",
  "mlflow.experiment.chat-session.metrics.latency-tag": "",
  "mlflow.experiment.chat-session.metrics.persona-tag": "",
  "mlflow.experiment.chat-session.metrics.persona-tooltip": "",
  "mlflow.experiment.chat-session.metrics.tokens-tag": "",
  "mlflow.experiment.chat-session.view-trace": "",
  "mlflow.experiment.evaluations.ai-judge-tag": "",
  "mlflow.experiment.evaluations.human-judge-tag": "",
  "mlflow.experiment.list.tag.add": "",
  "mlflow.experiment.overview": "",
  "mlflow.experiment.overview.detect-issues-button": "",
  "mlflow.experiment.overview.filestore-warning": "",
  "mlflow.experiment.overview.tabs": "",
  "mlflow.experiment.overview.time-unit-selector": "",
  "mlflow.experiment.prompt.optimize-modal": "",
  "mlflow.experiment.prompt.optimize-modal.mlflow-link": "",
  "mlflow.experiment.trace_location_path.button": "",
  "mlflow.experiment.trace_location_path.tooltip": "",

  // -- mlflow.experiment-evaluation-monitoring --
  "mlflow.experiment-evaluation-monitoring.date-selector": "",
  "mlflow.experiment-evaluation-monitoring.date-selector-button": "",
  "mlflow.experiment-evaluation-monitoring.evals-logs-table-cell": "",
  "mlflow.experiment-evaluation-monitoring.evals-logs-table-cell-tooltip": "",
  "mlflow.experiment-evaluation-monitoring.evals-logs-table-cell.spacer": "",
  "mlflow.experiment-evaluation-monitoring.evals-logs-table-header-select-cell": "",
  "mlflow.experiment-evaluation-monitoring.trace-info-hover-other-request-time": "",
  "mlflow.experiment-evaluation-monitoring.trace-info-hover-request-time": "",

  // -- mlflow.experiment-page --
  "mlflow.experiment-page.header.back-icon-button": "",
  "mlflow.experiment-page.header.docs-link": "",
  "mlflow.experiment-page.header.docs-link-button": "",

  // -- mlflow.experiment-scorers --
  "mlflow.experiment-scorers.add-variable-button": "",
  "mlflow.experiment-scorers.add-variable-conversation": "",
  "mlflow.experiment-scorers.add-variable-expectations": "",
  "mlflow.experiment-scorers.add-variable-inputs": "",
  "mlflow.experiment-scorers.add-variable-outputs": "",
  "mlflow.experiment-scorers.add-variable-trace": "",
  "mlflow.experiment-scorers.built-in-scorer-select": "",
  "mlflow.experiment-scorers.categorical-options-input": "",
  "mlflow.experiment-scorers.dict-value-type-select": "",
  "mlflow.experiment-scorers.documentation-link": "",
  "mlflow.experiment-scorers.empty-state-add-custom-code-scorer-button": "",
  "mlflow.experiment-scorers.empty-state-add-llm-scorer-button": "",
  "mlflow.experiment-scorers.form.scope-select": "",
  "mlflow.experiment-scorers.form.select-sessions-modal": "",
  "mlflow.experiment-scorers.form.select-sessions-modal.cancel": "",
  "mlflow.experiment-scorers.form.select-sessions-modal.ok": "",
  "mlflow.experiment-scorers.form.select-sessions-modal.ok-tooltip": "",
  "mlflow.experiment-scorers.form.select-traces-modal": "",
  "mlflow.experiment-scorers.form.select-traces-modal.cancel": "",
  "mlflow.experiment-scorers.form.select-traces-modal.ok": "",
  "mlflow.experiment-scorers.form.select-traces-modal.ok-tooltip": "",
  "mlflow.experiment-scorers.form.traces-picker.trigger": "",
  "mlflow.experiment-scorers.guidelines-learn-more-link": "",
  "mlflow.experiment-scorers.guidelines-text-area": "",
  "mlflow.experiment-scorers.instructions-learn-more-link": "",
  "mlflow.experiment-scorers.judges-error-banner": "",
  "mlflow.experiment-scorers.judges-running-banner": "",
  "mlflow.experiment-scorers.judges-success-banner": "",
  "mlflow.experiment-scorers.list-element-type-select": "",
  "mlflow.experiment-scorers.model-input": "",
  "mlflow.experiment-scorers.name-input": "",
  "mlflow.experiment-scorers.new-custom-code-scorer-menu-item": "",
  "mlflow.experiment-scorers.new-scorer-button": "",
  "mlflow.experiment-scorers.output-type-select": "",
  "mlflow.experiment-scorers.scorer-status-tag": "",
  "mlflow.experiment-scorers.switch-to-endpoint-link": "",
  "mlflow.experiment-scorers.switch-to-manual-link": "",
  "mlflow.experiment-scorers.traces-view-create-judge": "",
  "mlflow.experiment-scorers.traces-view-judge-error": "",
  "mlflow.experiment-scorers.traces-view-judge-llm": "",
  "mlflow.experiment-scorers.traces-view-judge-search": "",
  "mlflow.experiment-scorers.traces-view-judge-select-modal": "",
  "mlflow.experiment-scorers.traces-view-judge-template": "",
  "mlflow.experiment-scorers.traces-view-judge-type-filter": "",

  // -- mlflow.experiment-side-nav --
  "mlflow.experiment-side-nav.classic-ml.models": "",
  "mlflow.experiment-side-nav.classic-ml.runs": "",
  "mlflow.experiment-side-nav.classic-ml.traces": "",
  "mlflow.experiment-side-nav.genai.agent-versions": "",
  "mlflow.experiment-side-nav.genai.datasets": "",
  "mlflow.experiment-side-nav.genai.evaluation-runs": "",
  "mlflow.experiment-side-nav.genai.judges": "",
  "mlflow.experiment-side-nav.genai.overview": "",
  "mlflow.experiment-side-nav.genai.prompts": "",
  "mlflow.experiment-side-nav.genai.sessions": "",
  "mlflow.experiment-side-nav.genai.traces": "",
  "mlflow.experiment-side-nav.genai.training-runs": "",

  // -- mlflow.experiment-sidebar --
  "mlflow.experiment-sidebar.back-button": "",

  // -- mlflow.experiment-tracking --
  "mlflow.experiment-tracking.evaluation-artifact-compare.run-header": "",
  "mlflow.experiment-tracking.evaluation-cell.evaluate-all": "",
  "mlflow.experiment-tracking.evaluation-cell.not-evaluable": "",
  "mlflow.experiment-tracking.evaluation-group-header.toggle": "",
  "mlflow.experiment-tracking.evaluation-prompt-output.add": "",
  "mlflow.experiment-tracking.evaluation-prompt-output.evaluate": "",
  "mlflow.experiment-tracking.evaluation-prompt-params.help": "",
  "mlflow.experiment-tracking.evaluation-table-actions.add-row": "",
  "mlflow.experiment-tracking.evaluation-table-column.toggle-detail": "",
  "mlflow.experiment-tracking.experiment-description.edit": "",
  "mlflow.experiment-tracking.metrics-plot-controls.reset": "",
  "mlflow.experiment-tracking.metrics-plot-controls.save": "",
  "mlflow.experiment-tracking.models-cell.model-link": "",
  "mlflow.experiment-tracking.models-header.info": "",
  "mlflow.experiment-tracking.run-description.display": "",
  "mlflow.experiment-tracking.run-source.branch": "",
  "mlflow.experiment-tracking.runs-filters.clear-1": "",
  "mlflow.experiment-tracking.runs-filters.toggle-sidepane": "",
  "mlflow.experiment-tracking.runs-group-selector.aggregation": "",

  // -- mlflow.experiment_list --
  "mlflow.experiment_list.demo_badge": "",
  "mlflow.experiment_list.demo_tooltip": "",

  // -- mlflow.experiment_list_table --
  "mlflow.experiment_list_table.create_experiment": "",

  // -- mlflow.experiment_list_view --
  "mlflow.experiment_list_view.bulk_delete_button": "",
  "mlflow.experiment_list_view.check_all_box": "",
  "mlflow.experiment_list_view.check_box": "",
  "mlflow.experiment_list_view.compare_experiments_button": "",
  "mlflow.experiment_list_view.error": "",
  "mlflow.experiment_list_view.max_traces.tooltip": "",
  "mlflow.experiment_list_view.new_experiment_button": "",
  "mlflow.experiment_list_view.pagination": "",
  "mlflow.experiment_list_view.sampled_badge.tooltip": "",
  "mlflow.experiment_list_view.search": "",
  "mlflow.experiment_list_view.table.header": "",
  "mlflow.experiment_list_view.tag_filter": "",
  "mlflow.experiment_list_view.tag_filter.add_filter_button": "",
  "mlflow.experiment_list_view.tag_filter.apply_filters_button": "",
  "mlflow.experiment_list_view.tag_filter.clear_filters_button": "",
  "mlflow.experiment_list_view.tag_filter.trigger": "",

  // -- mlflow.experiment_page --
  "mlflow.experiment_page.grouped_runs.open_runs_in_new_tab": "",
  "mlflow.experiment_page.mode.artifact": "",
  "mlflow.experiment_page.runs.add_new_tag": "",
  "mlflow.experiment_page.runs.add_tags": "",
  "mlflow.experiment_page.scorers.advanced_settings_toggle": "",
  "mlflow.experiment_page.scorers.auto_evaluate_toggle": "",
  "mlflow.experiment_page.scorers.filter_string_input": "",
  "mlflow.experiment_page.scorers.filter_string_syntax_link": "",
  "mlflow.experiment_page.scorers.search_traces_syntax_link": "",
  "mlflow.experiment_page.sort_dropdown.search": "",
  "mlflow.experiment_page.sort_dropdown.sort_asc": "",
  "mlflow.experiment_page.sort_dropdown.sort_desc": "",
  "mlflow.experiment_page.sort_dropdown.sort_option": "",
  "mlflow.experiment_page.sort_select_v2.sort_asc": "",
  "mlflow.experiment_page.sort_select_v2.sort_desc": "",
  "mlflow.experiment_page.sort_select_v2.toggle": "",
  "mlflow.experiment_page.table_resizer.collapse": "",

  // -- mlflow.experiment_side_nav --
  "mlflow.experiment_side_nav.assistant_beta_tag": "",
  "mlflow.experiment_side_nav.assistant_button": "",
  "mlflow.experiment_side_nav.assistant_tooltip": "",

  // -- mlflow.experiment_tracking --
  "mlflow.experiment_tracking.artifacts.logged_model_fallback_link": "",
  "mlflow.experiment_tracking.artifacts.model_version_link": "",
  "mlflow.experiment_tracking.charts.tooltip_run_link": "",
  "mlflow.experiment_tracking.common.line_smooth_slider": "",
  "mlflow.experiment_tracking.compare_header.experiments_breadcrumb_link": "",
  "mlflow.experiment_tracking.compare_runs.compare_experiments_link": "",
  "mlflow.experiment_tracking.compare_runs.experiment_link": "",
  "mlflow.experiment_tracking.compare_runs.experiment_name_link": "",
  "mlflow.experiment_tracking.compare_runs.metric_chart_link": "",
  "mlflow.experiment_tracking.compare_runs.run_uuid_link": "",
  "mlflow.experiment_tracking.dataset_drawer.run_link": "",
  "mlflow.experiment_tracking.evaluation.run_header_link": "",
  "mlflow.experiment_tracking.evaluation_datasets.dataset_link": "",
  "mlflow.experiment_tracking.evaluation_runs.model_version_link": "",
  "mlflow.experiment_tracking.evaluation_runs.run_link": "",
  "mlflow.experiment_tracking.experiment_list.demo_experiment_link": "",
  "mlflow.experiment_tracking.experiment_list.experiment_name_link": "",
  "mlflow.experiment_tracking.header.experiment_name_breadcrumb_link": "",
  "mlflow.experiment_tracking.header.experiments_breadcrumb_link": "",
  "mlflow.experiment_tracking.issue_detection.breadcrumb_evaluation_runs_link": "",
  "mlflow.experiment_tracking.issue_detection.breadcrumb_experiment_link": "",
  "mlflow.experiment_tracking.issue_detection.breadcrumb_experiments_link": "",
  "mlflow.experiment_tracking.linked_prompts.prompt_name_link": "",
  "mlflow.experiment_tracking.linked_prompts.prompt_version_link": "",
  "mlflow.experiment_tracking.metric_view.compare_experiments_link": "",
  "mlflow.experiment_tracking.metric_view.compare_runs_link": "",
  "mlflow.experiment_tracking.metric_view.experiment_link": "",
  "mlflow.experiment_tracking.metric_view.multiple_experiments_link": "",
  "mlflow.experiment_tracking.metric_view.run_link": "",
  "mlflow.experiment_tracking.metrics_summary.run_link": "",
  "mlflow.experiment_tracking.run_links.run_link": "",
  "mlflow.experiment_tracking.runs_table.experiment_name_link": "",
  "mlflow.experiment_tracking.runs_table.group_parent_link": "",
  "mlflow.experiment_tracking.runs_table.logged_model_tooltip_link": "",
  "mlflow.experiment_tracking.runs_table.logged_model_v3_link": "",
  "mlflow.experiment_tracking.runs_table.model_version_link": "",
  "mlflow.experiment_tracking.runs_table.run_name_link": "",
  "mlflow.experiment_tracking.side_nav.section_item_link": "",

  // -- mlflow.experiment_view --
  "mlflow.experiment_view.header.experiment-name-tooltip": "",
  "mlflow.experiment_view.header.experiment_kind_inference_modal": "",
  "mlflow.experiment_view.header.experiment_kind_inference_popover": "",
  "mlflow.experiment_view.header.experiment_kind_inference_popover.confirm": "",
  "mlflow.experiment_view.header.experiment_kind_inference_popover.dismiss": "",
  "mlflow.experiment_view.header.experiment_kind_selector": "",
  "mlflow.experiment_view.header.experiment_kind_selector.tooltip": "",

  // -- mlflow.experiment_view_runs_table --
  "mlflow.experiment_view_runs_table.column_header.models.tooltip": "",

  // -- mlflow.export-traces-to-dataset-modal --
  "mlflow.export-traces-to-dataset-modal": "",
  "mlflow.export-traces-to-dataset-modal.header-checkbox": "",
  "mlflow.export-traces-to-dataset-modal.multiturn-error": "",
  "mlflow.export-traces-to-dataset-modal.row-checkbox": "",

  // -- mlflow.gateway --
  "mlflow.gateway.api-key-details.drawer": "",
  "mlflow.gateway.api-key-details.drawer.cancel-button": "",
  "mlflow.gateway.api-key-details.drawer.edit": "",
  "mlflow.gateway.api-key-details.drawer.edit-button": "",
  "mlflow.gateway.api-key-details.drawer.edit-error": "",
  "mlflow.gateway.api-key-details.drawer.edit-name": "",
  "mlflow.gateway.api-key-details.drawer.edit-provider": "",
  "mlflow.gateway.api-key-details.drawer.name-tooltip": "",
  "mlflow.gateway.api-key-details.drawer.provider-tooltip": "",
  "mlflow.gateway.api-key-details.drawer.save-button": "",
  "mlflow.gateway.api-keys.bulk-delete-button": "",
  "mlflow.gateway.api-keys.columns-button": "",
  "mlflow.gateway.api-keys.columns-dropdown": "",
  "mlflow.gateway.api-keys.create-button": "",
  "mlflow.gateway.api-keys.created-header": "",
  "mlflow.gateway.api-keys.endpoints-header": "",
  "mlflow.gateway.api-keys.error": "",
  "mlflow.gateway.api-keys.filter": "",
  "mlflow.gateway.api-keys.list.endpoints-link": "",
  "mlflow.gateway.api-keys.list.row": "",
  "mlflow.gateway.api-keys.list.used-by-link": "",
  "mlflow.gateway.api-keys.name-header": "",
  "mlflow.gateway.api-keys.provider-header": "",
  "mlflow.gateway.api-keys.row-checkbox": "",
  "mlflow.gateway.api-keys.search": "",
  "mlflow.gateway.api-keys.select-all-checkbox": "",
  "mlflow.gateway.api-keys.updated-header": "",
  "mlflow.gateway.api-keys.used-by-header": "",
  "mlflow.gateway.api_keys.binding_endpoint_link": "",
  "mlflow.gateway.api_keys.endpoint_link": "",
  "mlflow.gateway.bindings-using-key.drawer": "",
  "mlflow.gateway.budgets-list.action-header": "",
  "mlflow.gateway.budgets-list.actions-header": "",
  "mlflow.gateway.budgets-list.budget-amount-tooltip": "",
  "mlflow.gateway.budgets-list.budget-exceeded-tooltip": "",
  "mlflow.gateway.budgets-list.current-spend-header": "",
  "mlflow.gateway.budgets-list.current-spend-tooltip": "",
  "mlflow.gateway.budgets-list.delete-button": "",
  "mlflow.gateway.budgets-list.duration-header": "",
  "mlflow.gateway.budgets-list.edit-button": "",
  "mlflow.gateway.budgets-list.limit-header": "",
  "mlflow.gateway.budgets-list.next-page": "",
  "mlflow.gateway.budgets-list.previous-page": "",
  "mlflow.gateway.budgets-list.updated-header": "",
  "mlflow.gateway.budgets-list.window-end-header": "",
  "mlflow.gateway.budgets-list.window-end-tooltip": "",
  "mlflow.gateway.budgets-list.window-start-header": "",
  "mlflow.gateway.budgets.breadcrumb_gateway_link": "",
  "mlflow.gateway.budgets.create-button": "",
  "mlflow.gateway.budgets.go_to_endpoints_link": "",
  "mlflow.gateway.budgets.tabs": "",
  "mlflow.gateway.bulk-delete-api-key-modal": "",
  "mlflow.gateway.bulk-delete-api-key-modal.cancel": "",
  "mlflow.gateway.bulk-delete-api-key-modal.delete": "",
  "mlflow.gateway.bulk-delete-api-key-modal.error": "",
  "mlflow.gateway.bulk-delete-api-key-modal.warning": "",
  "mlflow.gateway.create-api-key-modal": "",
  "mlflow.gateway.create-api-key-modal.error": "",
  "mlflow.gateway.create-api-key-modal.provider": "",
  "mlflow.gateway.create-budget-policy-modal": "",
  "mlflow.gateway.create-budget-policy-modal.alert-webhook-info": "",
  "mlflow.gateway.create-budget-policy-modal.budget-amount": "",
  "mlflow.gateway.create-budget-policy-modal.duration": "",
  "mlflow.gateway.create-budget-policy-modal.error": "",
  "mlflow.gateway.create-budget-policy-modal.on-exceeded": "",
  "mlflow.gateway.create-budget-policy-modal.reset-period-tooltip": "",
  "mlflow.gateway.create-endpoint-modal": "",
  "mlflow.gateway.create-endpoint.secret-select": "",
  "mlflow.gateway.create-endpoint.usage-tracking": "",
  "mlflow.gateway.create_endpoint.breadcrumb_endpoints_link": "",
  "mlflow.gateway.create_endpoint.breadcrumb_gateway_link": "",
  "mlflow.gateway.delete-api-key-modal": "",
  "mlflow.gateway.delete-budget-policy-modal": "",
  "mlflow.gateway.delete-endpoint-modal": "",
  "mlflow.gateway.delete-endpoint-modal.cancel": "",
  "mlflow.gateway.delete-endpoint-modal.delete": "",
  "mlflow.gateway.delete-endpoint-modal.error": "",
  "mlflow.gateway.delete-endpoint-modal.warning": "",
  "mlflow.gateway.edit-api-key-modal": "",
  "mlflow.gateway.edit-api-key-modal.error": "",
  "mlflow.gateway.edit-api-key-modal.name": "",
  "mlflow.gateway.edit-api-key-modal.name-tooltip": "",
  "mlflow.gateway.edit-api-key-modal.provider": "",
  "mlflow.gateway.edit-api-key-modal.provider-tooltip": "",
  "mlflow.gateway.edit-budget-policy-modal": "",
  "mlflow.gateway.edit-budget-policy-modal.budget-amount": "",
  "mlflow.gateway.edit-budget-policy-modal.duration": "",
  "mlflow.gateway.edit-budget-policy-modal.error": "",
  "mlflow.gateway.edit-budget-policy-modal.on-exceeded": "",
  "mlflow.gateway.edit-budget-policy-modal.on-exceeded-tooltip": "",
  "mlflow.gateway.edit-budget-policy-modal.reset-period-tooltip": "",
  "mlflow.gateway.edit-endpoint-name-modal": "",
  "mlflow.gateway.edit-endpoint-name-modal.error": "",
  "mlflow.gateway.edit-endpoint-name-modal.name-input": "",
  "mlflow.gateway.edit-endpoint.api-key-link": "",
  "mlflow.gateway.edit-endpoint.cancel": "",
  "mlflow.gateway.edit-endpoint.error": "",
  "mlflow.gateway.edit-endpoint.fallback": "",
  "mlflow.gateway.edit-endpoint.mutation-error": "",
  "mlflow.gateway.edit-endpoint.name-edit-button": "",
  "mlflow.gateway.edit-endpoint.name-edit-tooltip": "",
  "mlflow.gateway.edit-endpoint.save": "",
  "mlflow.gateway.edit-endpoint.save-tooltip": "",
  "mlflow.gateway.edit-endpoint.starter-code.api": "",
  "mlflow.gateway.edit-endpoint.starter-code.copy": "",
  "mlflow.gateway.edit-endpoint.starter-code.try-in-browser": "",
  "mlflow.gateway.edit-endpoint.traffic-split": "",
  "mlflow.gateway.edit-endpoint.try-it-modal": "",
  "mlflow.gateway.edit-endpoint.try-it-modal.request-tooltip": "",
  "mlflow.gateway.edit-endpoint.usage-tracking-info": "",
  "mlflow.gateway.edit-endpoint.usage-tracking.toggle": "",
  "mlflow.gateway.edit_endpoint.breadcrumb_endpoints_link": "",
  "mlflow.gateway.edit_endpoint.breadcrumb_gateway_link": "",
  "mlflow.gateway.edit_endpoint.traces_link": "",
  "mlflow.gateway.endpoint-bindings.accordion": "",
  "mlflow.gateway.endpoint-bindings.drawer": "",
  "mlflow.gateway.endpoint-usage-modal": "",
  "mlflow.gateway.endpoint.guardrails-tab-tooltip": "",
  "mlflow.gateway.endpoint.tabs": "",
  "mlflow.gateway.endpoint.traces-tab-tooltip": "",
  "mlflow.gateway.endpoint.usage-tab-tooltip": "",
  "mlflow.gateway.endpoint.usage.view-full-dashboard": "",
  "mlflow.gateway.endpoints-list": "",
  "mlflow.gateway.endpoints-list.bindings-header": "",
  "mlflow.gateway.endpoints-list.columns-button": "",
  "mlflow.gateway.endpoints-list.columns-dropdown": "",
  "mlflow.gateway.endpoints-list.create-link": "",
  "mlflow.gateway.endpoints-list.created-header": "",
  "mlflow.gateway.endpoints-list.delete-button": "",
  "mlflow.gateway.endpoints-list.duplicate-button": "",
  "mlflow.gateway.endpoints-list.duplicate-error": "",
  "mlflow.gateway.endpoints-list.models-header": "",
  "mlflow.gateway.endpoints-list.models-toggle": "",
  "mlflow.gateway.endpoints-list.modified-header": "",
  "mlflow.gateway.endpoints-list.name-header": "",
  "mlflow.gateway.endpoints-list.provider-header": "",
  "mlflow.gateway.endpoints-list.provider-tag": "",
  "mlflow.gateway.endpoints-list.provider-toggle": "",
  "mlflow.gateway.endpoints-list.row-checkbox": "",
  "mlflow.gateway.endpoints-list.search": "",
  "mlflow.gateway.endpoints-list.select-all-checkbox": "",
  "mlflow.gateway.endpoints-using-key.drawer": "",
  "mlflow.gateway.endpoints.breadcrumb_gateway_link": "",
  "mlflow.gateway.endpoints.create-button": "",
  "mlflow.gateway.endpoints.endpoint_name_link": "",
  "mlflow.gateway.guardrails.action-header": "",
  "mlflow.gateway.guardrails.action-option.sanitization": "",
  "mlflow.gateway.guardrails.action-option.validation": "",
  "mlflow.gateway.guardrails.action-tag": "",
  "mlflow.gateway.guardrails.add": "",
  "mlflow.gateway.guardrails.add-modal": "",
  "mlflow.gateway.guardrails.back": "",
  "mlflow.gateway.guardrails.bulk-remove-cancel": "",
  "mlflow.gateway.guardrails.bulk-remove-confirm": "",
  "mlflow.gateway.guardrails.bulk-remove-error": "",
  "mlflow.gateway.guardrails.bulk-remove-modal": "",
  "mlflow.gateway.guardrails.cancel": "",
  "mlflow.gateway.guardrails.config-instructions": "",
  "mlflow.gateway.guardrails.config-name": "",
  "mlflow.gateway.guardrails.create": "",
  "mlflow.gateway.guardrails.create-tooltip": "",
  "mlflow.gateway.guardrails.delete": "",
  "mlflow.gateway.guardrails.detail-cancel": "",
  "mlflow.gateway.guardrails.detail-delete": "",
  "mlflow.gateway.guardrails.detail-modal": "",
  "mlflow.gateway.guardrails.detail-prompt": "",
  "mlflow.gateway.guardrails.detail-save": "",
  "mlflow.gateway.guardrails.error": "",
  "mlflow.gateway.guardrails.name-header": "",
  "mlflow.gateway.guardrails.placement-header": "",
  "mlflow.gateway.guardrails.placement-popover": "",
  "mlflow.gateway.guardrails.row-checkbox": "",
  "mlflow.gateway.guardrails.search": "",
  "mlflow.gateway.guardrails.select-all": "",
  "mlflow.gateway.guardrails.stage-tag": "",
  "mlflow.gateway.guardrails.type-card.custom": "",
  "mlflow.gateway.guardrails.type-card.pii": "",
  "mlflow.gateway.guardrails.type-card.safety": "",
  "mlflow.gateway.model-select.capability": "",
  "mlflow.gateway.model-selector-modal": "",
  "mlflow.gateway.model-selector-modal.cancel": "",
  "mlflow.gateway.model-selector-modal.confirm": "",
  "mlflow.gateway.model-selector-modal.custom-model": "",
  "mlflow.gateway.model-selector-modal.deprecation-tooltip": "",
  "mlflow.gateway.model-selector-modal.filter-button": "",
  "mlflow.gateway.model-selector-modal.filter-popover": "",
  "mlflow.gateway.model-selector-modal.filter.promptCaching": "",
  "mlflow.gateway.model-selector-modal.filter.reasoning": "",
  "mlflow.gateway.model-selector-modal.filter.structuredOutput": "",
  "mlflow.gateway.model-selector-modal.filter.tools": "",
  "mlflow.gateway.model-selector-modal.radio-group": "",
  "mlflow.gateway.model-selector-modal.search": "",
  "mlflow.gateway.quick_start.anthropic": "",
  "mlflow.gateway.quick_start.browse_all": "",
  "mlflow.gateway.quick_start.browse_all.button": "",
  "mlflow.gateway.quick_start.compact.anthropic": "",
  "mlflow.gateway.quick_start.compact.browse_all": "",
  "mlflow.gateway.quick_start.compact.databricks": "",
  "mlflow.gateway.quick_start.compact.gemini": "",
  "mlflow.gateway.quick_start.compact.openai": "",
  "mlflow.gateway.quick_start.databricks": "",
  "mlflow.gateway.quick_start.gemini": "",
  "mlflow.gateway.quick_start.openai": "",
  "mlflow.gateway.setup.install.copy": "",
  "mlflow.gateway.setup.passphrase.copy": "",
  "mlflow.gateway.setup.passphrase.warning": "",
  "mlflow.gateway.setup.server.copy": "",
  "mlflow.gateway.setup_guide": "",
  "mlflow.gateway.side-nav.budgets.tooltip": "",
  "mlflow.gateway.side-nav.endpoints.tooltip": "",
  "mlflow.gateway.side-nav.usage.tooltip": "",
  "mlflow.gateway.side_nav.tab_link": "",
  "mlflow.gateway.usage-modal.copy": "",
  "mlflow.gateway.usage-modal.passthrough-view-mode": "",
  "mlflow.gateway.usage-modal.tabs": "",
  "mlflow.gateway.usage-modal.try-it.provider": "",
  "mlflow.gateway.usage-modal.try-it.request": "",
  "mlflow.gateway.usage-modal.try-it.request-tooltip": "",
  "mlflow.gateway.usage-modal.try-it.request-tooltip-passthrough": "",
  "mlflow.gateway.usage-modal.try-it.reset": "",
  "mlflow.gateway.usage-modal.try-it.response": "",
  "mlflow.gateway.usage-modal.try-it.response-tooltip": "",
  "mlflow.gateway.usage-modal.try-it.send": "",
  "mlflow.gateway.usage-modal.try-it.unified-variant": "",
  "mlflow.gateway.usage-modal.unified-view-mode": "",
  "mlflow.gateway.usage.breadcrumb_gateway_link": "",
  "mlflow.gateway.usage.endpoint-selector": "",
  "mlflow.gateway.usage.go_to_endpoints_link": "",
  "mlflow.gateway.usage.go_to_endpoints_link_logs": "",
  "mlflow.gateway.usage.tabs": "",
  "mlflow.gateway.usage.user-selector": "",

  // -- mlflow.genai-traces-table --
  "mlflow.genai-traces-table.actions-disabled-tooltip": "",
  "mlflow.genai-traces-table.actions-dropdown": "",
  "mlflow.genai-traces-table.assessment-cell-judge-running": "",
  "mlflow.genai-traces-table.average-values-tag": "",
  "mlflow.genai-traces-table.chat_sessions_table.session_row_link": "",
  "mlflow.genai-traces-table.compare-traces": "",
  "mlflow.genai-traces-table.delete-session": "",
  "mlflow.genai-traces-table.delete-traces": "",
  "mlflow.genai-traces-table.edit-tags": "",
  "mlflow.genai-traces-table.execution-time": "",
  "mlflow.genai-traces-table.export-to-datasets": "",
  "mlflow.genai-traces-table.issue-tag": "",
  "mlflow.genai-traces-table.issue-tag-overflow-trigger": "",
  "mlflow.genai-traces-table.logged_model_cell.model_link": "",
  "mlflow.genai-traces-table.prompt_link": "",
  "mlflow.genai-traces-table.run-judges": "",
  "mlflow.genai-traces-table.run_name_link": "",
  "mlflow.genai-traces-table.session": "",
  "mlflow.genai-traces-table.session-header-request-time": "",
  "mlflow.genai-traces-table.session-header-request-time-other": "",
  "mlflow.genai-traces-table.session-header-session-id": "",
  "mlflow.genai-traces-table.session-header-session-id-other": "",
  "mlflow.genai-traces-table.session-header.pass-fail-aggregated-tooltip": "",
  "mlflow.genai-traces-table.session-header.select-cell": "",
  "mlflow.genai-traces-table.session-header.toggle-expanded": "",
  "mlflow.genai-traces-table.session-numeric-assessment": "",
  "mlflow.genai-traces-table.session-string-tag": "",
  "mlflow.genai-traces-table.session-tokens": "",
  "mlflow.genai-traces-table.session_id_link": "",
  "mlflow.genai-traces-table.status": "",
  "mlflow.genai-traces-table.tag_view_modal.tag_value_copy_button": "",
  "mlflow.genai-traces-table.tokens": "",
  "mlflow.genai-traces-table.trace-id": "",

  // -- mlflow.genai_traces_table --
  "mlflow.genai_traces_table.filter_dropdown": "",
  "mlflow.genai_traces_table.sort_dropdown.no_results": "",
  "mlflow.genai_traces_table.sort_dropdown.search": "",
  "mlflow.genai_traces_table.sort_dropdown.sort_desc": "",
  "mlflow.genai_traces_table.sort_dropdown.sort_option": "",

  // -- mlflow.genai_traces_table_filter --
  "mlflow.genai_traces_table_filter.filter_dropdown": "",

  // -- mlflow.home --
  "mlflow.home.create_workspace_modal": "",
  "mlflow.home.create_workspace_modal.error": "",
  "mlflow.home.create_workspace_modal.workspace_artifact_root_input": "",
  "mlflow.home.create_workspace_modal.workspace_description_input": "",
  "mlflow.home.create_workspace_modal.workspace_name_input": "",
  "mlflow.home.demo-banner.launch": "",
  "mlflow.home.experiments.create": "",
  "mlflow.home.experiments.error": "",
  "mlflow.home.experiments.retry": "",
  "mlflow.home.experiments.view_all_link": "",
  "mlflow.home.feature_card.ai_gateway": "",
  "mlflow.home.feature_card.evaluation": "",
  "mlflow.home.feature_card.experiments": "",
  "mlflow.home.feature_card.prompts": "",
  "mlflow.home.feature_card.tracing": "",
  "mlflow.home.log_traces.drawer": "",
  "mlflow.home.log_traces.drawer.configure.copy": "",
  "mlflow.home.log_traces.drawer.select-framework": "",
  "mlflow.home.log_traces.experiments_link": "",
  "mlflow.home.news.agents_as_a_judge": "",
  "mlflow.home.news.auto_tune_llm_judge": "",
  "mlflow.home.news.dataset_tracking": "",
  "mlflow.home.news.optimize_prompts": "",
  "mlflow.home.news.view_more": "",
  "mlflow.home.quick_action.gateway": "",
  "mlflow.home.quick_action.log_traces": "",
  "mlflow.home.quick_action.register_prompts": "",
  "mlflow.home.quick_action.run_evaluation": "",
  "mlflow.home.quick_action.train_models": "",
  "mlflow.home.telemetry-alert": "",
  "mlflow.home.workspaces.create": "",
  "mlflow.home.workspaces.create_button": "",
  "mlflow.home.workspaces.edit_artifact_root": "",
  "mlflow.home.workspaces.edit_description": "",
  "mlflow.home.workspaces.edit_input": "",
  "mlflow.home.workspaces.edit_modal": "",
  "mlflow.home.workspaces.error": "",
  "mlflow.home.workspaces.last_used_badge": "",
  "mlflow.home.workspaces.pagination": "",
  "mlflow.home.workspaces.retry": "",
  "mlflow.home.workspaces.workspace_link": "",
  "mlflow.home.workspaces_table.header.artifact_root": "",
  "mlflow.home.workspaces_table.header.description": "",
  "mlflow.home.workspaces_table.header.name": "",

  // -- mlflow.issue-detection --
  "mlflow.issue-detection.category-tag": "",
  "mlflow.issue-detection.completed": "",
  "mlflow.issue-detection.endpoint-link": "",

  // -- mlflow.issues --
  "mlflow.issues.cancel-button": "",
  "mlflow.issues.category-tag": "",
  "mlflow.issues.description-textarea": "",
  "mlflow.issues.edit-button": "",
  "mlflow.issues.issue-card": "",
  "mlflow.issues.move-to-pending-button": "",
  "mlflow.issues.reject-button": "",
  "mlflow.issues.resolve-button": "",
  "mlflow.issues.save-button": "",
  "mlflow.issues.severity-select": "",
  "mlflow.issues.severity-tag": "",
  "mlflow.issues.status-filter": "",
  "mlflow.issues.status-tag": "",

  // -- mlflow.legacy_compare_run --
  "mlflow.legacy_compare_run.run_id": "",
  "mlflow.legacy_compare_run.run_name": "",
  "mlflow.legacy_compare_run.time_row": "",

  // -- mlflow.logged_model --
  "mlflow.logged_model.dataset": "",
  "mlflow.logged_model.details.delete_button": "",
  "mlflow.logged_model.details.delete_modal": "",
  "mlflow.logged_model.details.delete_modal.error": "",
  "mlflow.logged_model.details.experiment-error": "",
  "mlflow.logged_model.details.linked_prompts.table.header": "",
  "mlflow.logged_model.details.metrics.table.header": "",
  "mlflow.logged_model.details.metrics.table.search": "",
  "mlflow.logged_model.details.more_actions": "",
  "mlflow.logged_model.details.not_registered_tag": "",
  "mlflow.logged_model.details.registered_model_version_tag": "",
  "mlflow.logged_model.details.related_runs.error": "",
  "mlflow.logged_model.details.runs.table.header": "",
  "mlflow.logged_model.details.runs.table.search": "",
  "mlflow.logged_model.details.source.branch": "",
  "mlflow.logged_model.details.source.branch_tooltip": "",
  "mlflow.logged_model.details.source.commit_hash": "",
  "mlflow.logged_model.details.source.commit_hash_popover": "",
  "mlflow.logged_model.details.user-action-error": "",
  "mlflow.logged_model.list.charts.search": "",
  "mlflow.logged_model.list.columns": "",
  "mlflow.logged_model.list.group_by": "",
  "mlflow.logged_model.list.group_by.none": "",
  "mlflow.logged_model.list.group_by.runs": "",
  "mlflow.logged_model.list.header.error": "",
  "mlflow.logged_model.list.metric_by_dataset_column_header": "",
  "mlflow.logged_model.list.order_by": "",
  "mlflow.logged_model.list.order_by.button_asc": "",
  "mlflow.logged_model.list.order_by.button_desc": "",
  "mlflow.logged_model.list.order_by.column_toggle": "",
  "mlflow.logged_model.list.order_by.filter": "",
  "mlflow.logged_model.list.registered_model_cell_version_tag": "",
  "mlflow.logged_model.list.sort": "",
  "mlflow.logged_model.list.view-mode": "",
  "mlflow.logged_model.list.view-mode-chart-tooltip": "",
  "mlflow.logged_model.list.view-mode-table-tooltip": "",
  "mlflow.logged_model.list_page.datasets_filter": "",
  "mlflow.logged_model.list_page.datasets_filter.toggle": "",
  "mlflow.logged_model.list_page.global_row_visibility_toggle": "",
  "mlflow.logged_model.list_page.global_row_visibility_toggle.options": "",
  "mlflow.logged_model.list_page.row_visibility_toggle": "",
  "mlflow.logged_model.name_cell_tooltip": "",
  "mlflow.logged_model.name_cell_version_tag": "",
  "mlflow.logged_model.status": "",
  "mlflow.logged_model.traces.traces_table.quickstart_docs_link": "",
  "mlflow.logged_model.traces.traces_table.set_active_model_quickstart_snippet_copy": "",

  // -- mlflow.logged_model_table --
  "mlflow.logged_model_table.group_toggle": "",

  // -- mlflow.logged_models --
  "mlflow.logged_models.details.description.edit": "",
  "mlflow.logged_models.details.model_version_link": "",
  "mlflow.logged_models.details_header.experiment_link": "",
  "mlflow.logged_models.details_header.models_tab_link": "",
  "mlflow.logged_models.details_metadata.source_run_id_link": "",
  "mlflow.logged_models.details_metadata.source_run_name_link": "",
  "mlflow.logged_models.details_nav.artifacts_link": "",
  "mlflow.logged_models.details_nav.overview_link": "",
  "mlflow.logged_models.details_nav.traces_link": "",
  "mlflow.logged_models.details_overview.source_run_link": "",
  "mlflow.logged_models.details_table.run_cell_link": "",
  "mlflow.logged_models.list.error": "",
  "mlflow.logged_models.list.example_code_modal": "",
  "mlflow.logged_models.list.genai_no_results_learn_more": "",
  "mlflow.logged_models.list.load_more": "",
  "mlflow.logged_models.list.ml_no_results_learn_more": "",
  "mlflow.logged_models.list.no_results_learn_more": "",
  "mlflow.logged_models.list.show_example_code": "",
  "mlflow.logged_models.table.group_source_run_link": "",
  "mlflow.logged_models.table.model_name_link": "",
  "mlflow.logged_models.table.model_version_link": "",
  "mlflow.logged_models.table.name_link": "",
  "mlflow.logged_models.table.original_model_tooltip_link": "",
  "mlflow.logged_models.table.registered_model_link": "",
  "mlflow.logged_models.table.source_run_link": "",

  // -- mlflow.model-registry --
  "mlflow.model-registry.model-list.model-name.tooltip": "",
  "mlflow.model-registry.model-list.model-tag.tooltip": "",
  "mlflow.model-registry.model-view.model-versions.version-status.tooltip": "",

  // -- mlflow.model-trace-explorer --
  "mlflow.model-trace-explorer.add-human-feedback": "",
  "mlflow.model-trace-explorer.run-judge": "",
  "mlflow.model-trace-explorer.session-id-tag": "",

  // -- mlflow.model_registry --
  "mlflow.model_registry.aliases.overflow_version_link": "",
  "mlflow.model_registry.aliases.version_link": "",
  "mlflow.model_registry.compare_versions.metric_link": "",
  "mlflow.model_registry.compare_versions.model_name_link": "",
  "mlflow.model_registry.compare_versions.registered_models_link": "",
  "mlflow.model_registry.compare_versions.run_uuid_link": "",
  "mlflow.model_registry.compare_versions.version_link": "",
  "mlflow.model_registry.model_list.model_name_link": "",
  "mlflow.model_registry.model_list.version_link": "",
  "mlflow.model_registry.model_view.breadcrumb_registered_models_link": "",
  "mlflow.model_registry.stage_transition_modal_v2": "",
  "mlflow.model_registry.stage_transition_modal_v2.archive_existing_versions": "",
  "mlflow.model_registry.stage_transition_modal_v2.archive_existing_versions.tooltip": "",
  "mlflow.model_registry.stage_transition_modal_v2.comment": "",
  "mlflow.model_registry.version_table.version_link": "",
  "mlflow.model_registry.version_view.breadcrumb_model_link": "",
  "mlflow.model_registry.version_view.breadcrumb_registered_models_link": "",
  "mlflow.model_registry.version_view.copied_from_link": "",
  "mlflow.model_registry.version_view.source_run_link": "",

  // -- mlflow.model_trace_explorer --
  "mlflow.model_trace_explorer.feedback_item.judge_trace_link": "",
  "mlflow.model_trace_explorer.header.session_id_link": "",
  "mlflow.model_trace_explorer.header_details.tag-session-id": "",
  "mlflow.model_trace_explorer.linked_prompts.prompt_link": "",
  "mlflow.model_trace_explorer.timeline.gateway_trace_link": "",

  // -- mlflow.node-level-metric-charts --
  "mlflow.node-level-metric-charts.filter.by_gpu": "",
  "mlflow.node-level-metric-charts.filter.by_node": "",
  "mlflow.node-level-metric-charts.filter.clear": "",
  "mlflow.node-level-metric-charts.filter.trigger": "",

  // -- mlflow.notebook --
  "mlflow.notebook.pagination": "",
  "mlflow.notebook.trace-ui-info": "",
  "mlflow.notebook.trace-ui-learn-more-link": "",
  "mlflow.notebook.trace-ui-see-in-mlflow-link": "",

  // -- mlflow.overview --
  "mlflow.overview.quality.assessment.view_traces_link": "",
  "mlflow.overview.quality.assessment_timeseries.view_traces_link": "",
  "mlflow.overview.quality.quality_summary_table": "",
  "mlflow.overview.quality_tab.empty_state.example_code_copy": "",
  "mlflow.overview.tools.error_rate.view_traces_link": "",
  "mlflow.overview.usage.errors.view_traces_link": "",
  "mlflow.overview.usage.latency.view_traces_link": "",
  "mlflow.overview.usage.token_stats.view_traces_link": "",
  "mlflow.overview.usage.token_usage.view_traces_link": "",
  "mlflow.overview.usage.trace_cost_over_time": "",
  "mlflow.overview.usage.trace_cost_over_time.dimension": "",
  "mlflow.overview.usage.trace_cost_over_time.item_selector": "",
  "mlflow.overview.usage.traces.view_traces_link": "",

  // -- mlflow.prompts --
  "mlflow.prompts.chat_creator.add_after": "",
  "mlflow.prompts.chat_creator.content": "",
  "mlflow.prompts.chat_creator.remove": "",
  "mlflow.prompts.chat_creator.role": "",
  "mlflow.prompts.compare.markdown-diff-warning": "",
  "mlflow.prompts.compare.toggle-markdown-rendering": "",
  "mlflow.prompts.create.commit_message": "",
  "mlflow.prompts.create.content": "",
  "mlflow.prompts.create.error": "",
  "mlflow.prompts.create.modal": "",
  "mlflow.prompts.create.name": "",
  "mlflow.prompts.create.response_format": "",
  "mlflow.prompts.create.toggle_advanced_settings": "",
  "mlflow.prompts.delete_modal": "",
  "mlflow.prompts.delete_version_modal": "",
  "mlflow.prompts.details.actions": "",
  "mlflow.prompts.details.actions.delete": "",
  "mlflow.prompts.details.breadcrumb_link": "",
  "mlflow.prompts.details.create": "",
  "mlflow.prompts.details.delete_version": "",
  "mlflow.prompts.details.markdown-rendering-tooltip": "",
  "mlflow.prompts.details.mode": "",
  "mlflow.prompts.details.plaintext-rendering-tooltip": "",
  "mlflow.prompts.details.preview.optimize": "",
  "mlflow.prompts.details.preview.usage_example_modal": "",
  "mlflow.prompts.details.preview.use": "",
  "mlflow.prompts.details.runs.show_more": "",
  "mlflow.prompts.details.select_baseline.tooltip": "",
  "mlflow.prompts.details.select_compared.tooltip": "",
  "mlflow.prompts.details.switch_sides": "",
  "mlflow.prompts.details.switch_sides.tooltip": "",
  "mlflow.prompts.details.tags.edit": "",
  "mlflow.prompts.details.toggle-markdown-rendering": "",
  "mlflow.prompts.details.version.add_tags": "",
  "mlflow.prompts.details.version.edit_model_config": "",
  "mlflow.prompts.details.version.edit_tags": "",
  "mlflow.prompts.details.version.goto": "",
  "mlflow.prompts.details.version.tags.show_more": "",
  "mlflow.prompts.edit_model_config.error": "",
  "mlflow.prompts.edit_model_config.modal": "",
  "mlflow.prompts.list.prompt_name_link": "",
  "mlflow.prompts.list.table.create_prompt": "",
  "mlflow.prompts.list.table.learn_more_link": "",
  "mlflow.prompts.list.tag.add": "",
  "mlflow.prompts.model_config.frequencyPenalty": "",
  "mlflow.prompts.model_config.help": "",
  "mlflow.prompts.model_config.maxTokens": "",
  "mlflow.prompts.model_config.modelName": "",
  "mlflow.prompts.model_config.presencePenalty": "",
  "mlflow.prompts.model_config.provider": "",
  "mlflow.prompts.model_config.stopSequences": "",
  "mlflow.prompts.model_config.temperature": "",
  "mlflow.prompts.model_config.topK": "",
  "mlflow.prompts.model_config.topP": "",
  "mlflow.prompts.version_runs.run_link": "",
  "mlflow.prompts.versions-table.row": "",
  "mlflow.prompts.versions.table.header": "",

  // -- mlflow.quality_tab --
  "mlflow.quality_tab.empty_state.learn_more_link": "",

  // -- mlflow.run --
  "mlflow.run.artifact_view.create_run.tooltip": "",
  "mlflow.run.artifact_view.evaluate_all.tooltip": "",
  "mlflow.run.artifact_view.preview_close": "",
  "mlflow.run.artifact_view.preview_sidebar_toggle": "",
  "mlflow.run.artifact_view.table_settings": "",
  "mlflow.run.artifact_view.table_settings.tooltip": "",
  "mlflow.run.row_actions.pinning.tooltip": "",
  "mlflow.run.row_actions.visibility.tooltip": "",

  // -- mlflow.run-page --
  "mlflow.run-page.view-mode-switch": "",

  // -- mlflow.run-view --
  "mlflow.run-view.compare-button": "",
  "mlflow.run-view.compare-button.tooltip": "",

  // -- mlflow.run_details --
  "mlflow.run_details.header.register-model-button.tooltip": "",
  "mlflow.run_details.header.register_model_from_logged_model.button": "",
  "mlflow.run_details.header.register_model_from_logged_model.dropdown_menu_item": "",
  "mlflow.run_details.header.register_model_from_logged_model.dropdown_menu_item.view_model_button":
    "",
  "mlflow.run_details.overview.child_runs.load_more_button": "",
  "mlflow.run_details.overview.source.commit_hash": "",
  "mlflow.run_details.overview.source.commit_hash_popover": "",
  "mlflow.run_details.overview.tags.add_button": "",
  "mlflow.run_details.overview.tags.edit_button": "",
  "mlflow.run_details.overview.tags.edit_button.tooltip": "",

  // -- mlflow.run_page --
  "mlflow.run_page.header.compare_experiments_link": "",
  "mlflow.run_page.header.experiment_name_link": "",
  "mlflow.run_page.header.experiment_tab_link": "",
  "mlflow.run_page.header.register_model_v3_view_link": "",
  "mlflow.run_page.header.register_model_view_link": "",
  "mlflow.run_page.header.registered_model_version_link": "",
  "mlflow.run_page.header.view_registered_model_link": "",
  "mlflow.run_page.logged_model.list.error": "",
  "mlflow.run_page.overview.child_run_link": "",
  "mlflow.run_page.overview.experiment_id_link": "",
  "mlflow.run_page.overview.issue_detection_experiment_link": "",
  "mlflow.run_page.overview.logged_model_link": "",
  "mlflow.run_page.overview.logged_model_v3_link": "",
  "mlflow.run_page.overview.metric_chart_link": "",
  "mlflow.run_page.overview.metric_model_link": "",
  "mlflow.run_page.overview.parent_run_link": "",
  "mlflow.run_page.overview.registered_model_link": "",
  "mlflow.run_page.overview.registered_prompt_link": "",
  "mlflow.run_page.overview.user_link": "",

  // -- mlflow.runs_chart --
  "mlflow.runs_chart.tooltip.hide_run": "",
  "mlflow.runs_chart.tooltip.pin_run": "",

  // -- mlflow.schema_table --
  "mlflow.schema_table.header.name": "",
  "mlflow.schema_table.header.type": "",
  "mlflow.schema_table.search_input": "",

  // -- mlflow.settings --
  "mlflow.settings.demo.clear-all-button": "",
  "mlflow.settings.demo.confirm-modal": "",
  "mlflow.settings.general.preferences-card": "",
  "mlflow.settings.telemetry.documentation-link": "",
  "mlflow.settings.telemetry.toggle-switch": "",
  "mlflow.settings.theme.toggle-switch": "",
  "mlflow.settings.webhooks.create-button": "",
  "mlflow.settings.webhooks.delete-button": "",
  "mlflow.settings.webhooks.delete-modal": "",
  "mlflow.settings.webhooks.description-input": "",
  "mlflow.settings.webhooks.edit-button": "",
  "mlflow.settings.webhooks.error-alert": "",
  "mlflow.settings.webhooks.event-checkbox": "",
  "mlflow.settings.webhooks.form-error-alert": "",
  "mlflow.settings.webhooks.form-modal": "",
  "mlflow.settings.webhooks.name-input": "",
  "mlflow.settings.webhooks.status-switch": "",
  "mlflow.settings.webhooks.test-button": "",
  "mlflow.settings.webhooks.test-result-alert": "",
  "mlflow.settings.webhooks.url-input": "",

  // -- mlflow.shared --
  "mlflow.shared.copy_button": "",
  "mlflow.shared.copy_button.tooltip": "",

  // -- mlflow.sidebar --
  "mlflow.sidebar.account": "",
  "mlflow.sidebar.assistant_beta_tag": "",
  "mlflow.sidebar.assistant_button": "",
  "mlflow.sidebar.assistant_tooltip": "",
  "mlflow.sidebar.docs_link": "",
  "mlflow.sidebar.experiments_tab_link": "",
  "mlflow.sidebar.gateway_budgets_tab_link": "",
  "mlflow.sidebar.gateway_endpoints_tab_link": "",
  "mlflow.sidebar.gateway_new_tag": "",
  "mlflow.sidebar.gateway_tab_link": "",
  "mlflow.sidebar.gateway_usage_tab_link": "",
  "mlflow.sidebar.home_tab_link": "",
  "mlflow.sidebar.logo_home_link": "",
  "mlflow.sidebar.logout": "",
  "mlflow.sidebar.models_tab_link": "",
  "mlflow.sidebar.prompts_tab_link": "",
  "mlflow.sidebar.settings_back_link": "",
  "mlflow.sidebar.settings_general_link": "",
  "mlflow.sidebar.settings_llm_connections_link": "",
  "mlflow.sidebar.settings_tab_link": "",
  "mlflow.sidebar.settings_webhooks_link": "",
  "mlflow.sidebar.workflow_switch": "",
  "mlflow.sidebar.workflow_switch.tooltip": "",
  "mlflow.sidebar.workspace_home_link": "",

  // -- mlflow.storybook --
  "mlflow.storybook.country-selector": "",
  "mlflow.storybook.custom-render": "",
  "mlflow.storybook.empty-modal": "",
  "mlflow.storybook.grouped": "",
  "mlflow.storybook.hover-default": "",
  "mlflow.storybook.hover-table": "",
  "mlflow.storybook.hover-tertiary": "",
  "mlflow.storybook.model-selector": "",
  "mlflow.storybook.provider-selector": "",
  "mlflow.storybook.search": "",
  "mlflow.storybook.selector-modal": "",
  "mlflow.storybook.simple": "",
  "mlflow.storybook.with-description": "",

  // -- mlflow.tags_cell_renderer --
  "mlflow.tags_cell_renderer.traces_table.edit_tag": "",

  // -- mlflow.telemetry --
  "mlflow.telemetry.info_alert.documentation_link": "",

  // -- mlflow.traces --
  "mlflow.traces.empty_state_generic_quickstart.copy": "",
  "mlflow.traces.issue-detection-modal": "",
  "mlflow.traces.issue-detection-modal.cancel": "",
  "mlflow.traces.issue-detection-modal.category.adherence": "",
  "mlflow.traces.issue-detection-modal.category.correctness": "",
  "mlflow.traces.issue-detection-modal.category.execution": "",
  "mlflow.traces.issue-detection-modal.category.latency": "",
  "mlflow.traces.issue-detection-modal.category.relevance": "",
  "mlflow.traces.issue-detection-modal.category.safety": "",
  "mlflow.traces.issue-detection-modal.error": "",
  "mlflow.traces.issue-detection-modal.model": "",
  "mlflow.traces.issue-detection-modal.next": "",
  "mlflow.traces.issue-detection-modal.previous": "",
  "mlflow.traces.issue-detection-modal.select-traces": "",
  "mlflow.traces.issue-detection-modal.submit": "",
  "mlflow.traces.issue-detection.api-key.auth-mode-radio-group": "",
  "mlflow.traces.issue-detection.api-key.config-input": "",
  "mlflow.traces.issue-detection.api-key.mode": "",
  "mlflow.traces.issue-detection.api-key.secret-input": "",
  "mlflow.traces.issue-detection.cancel-button": "",
  "mlflow.traces.issue-detection.view-issues-button": "",
  "mlflow.traces.issue-detection.view-issues-link": "",
  "mlflow.traces.issue-detection.view-traces-link": "",

  // -- mlflow.traces-tab --
  "mlflow.traces-tab.trace-count": "",

  // -- mlflow.traces-table --
  "mlflow.traces-table.column-header-tooltip": "",
  "mlflow.traces-table.group-by-session-button": "",
  "mlflow.traces-table.group-by-session-button.tooltip": "",
  "mlflow.traces-table.refresh-button": "",
  "mlflow.traces-table.refresh-button.tooltip": "",

  // -- shared.media-rendering-utils --
  "shared.media-rendering-utils.fetch-download": "",

  // -- shared.model-trace-explorer --
  "shared.model-trace-explorer.add-expectation": "",
  "shared.model-trace-explorer.add-feedback": "",
  "shared.model-trace-explorer.add-feedback-in-group-tooltip": "",
  "shared.model-trace-explorer.add-new-assessment": "",
  "shared.model-trace-explorer.assesment-value-tag": "",
  "shared.model-trace-explorer.assesment-value-tooltip": "",
  "shared.model-trace-explorer.assessment-count": "",
  "shared.model-trace-explorer.assessment-create-button": "",
  "shared.model-trace-explorer.assessment-data-type-select": "",
  "shared.model-trace-explorer.assessment-delete-button": "",
  "shared.model-trace-explorer.assessment-delete-modal": "",
  "shared.model-trace-explorer.assessment-edit-button": "",
  "shared.model-trace-explorer.assessment-edit-cancel-button": "",
  "shared.model-trace-explorer.assessment-edit-data-type-select": "",
  "shared.model-trace-explorer.assessment-edit-rationale-input": "",
  "shared.model-trace-explorer.assessment-edit-save-button": "",
  "shared.model-trace-explorer.assessment-edit-value-boolean-input": "",
  "shared.model-trace-explorer.assessment-edit-value-number-input": "",
  "shared.model-trace-explorer.assessment-edit-value-string-input": "",
  "shared.model-trace-explorer.assessment-more-button": "",
  "shared.model-trace-explorer.assessment-name-typeahead": "",
  "shared.model-trace-explorer.assessment-notes-info-tooltip": "",
  "shared.model-trace-explorer.assessment-notes-input": "",
  "shared.model-trace-explorer.assessment-notes-save": "",
  "shared.model-trace-explorer.assessment-rationale-input": "",
  "shared.model-trace-explorer.assessment-source-name": "",
  "shared.model-trace-explorer.assessment-value-boolean-input": "",
  "shared.model-trace-explorer.assessment-value-number-input": "",
  "shared.model-trace-explorer.assessment-value-string-input": "",
  "shared.model-trace-explorer.assessments-pane-toggle": "",
  "shared.model-trace-explorer.attachment-audio-play": "",
  "shared.model-trace-explorer.attachment-download": "",
  "shared.model-trace-explorer.attachment-image-preview": "",
  "shared.model-trace-explorer.cancel-evaluation": "",
  "shared.model-trace-explorer.cancel-evaluation-in-group": "",
  "shared.model-trace-explorer.close-assessments-pane": "",
  "shared.model-trace-explorer.close-assessments-pane-tooltip": "",
  "shared.model-trace-explorer.compare-modal.trace-id-tag": "",
  "shared.model-trace-explorer.compare-modal.trace-id-tag-tooltip": "",
  "shared.model-trace-explorer.content-tab.render-mode": "",
  "shared.model-trace-explorer.conversation-toggle": "",
  "shared.model-trace-explorer.copy-snippet": "",
  "shared.model-trace-explorer.cost-hovercard.input-cost.tag": "",
  "shared.model-trace-explorer.cost-hovercard.output-cost.tag": "",
  "shared.model-trace-explorer.cost-hovercard.total-cost.tag": "",
  "shared.model-trace-explorer.expand": "",
  "shared.model-trace-explorer.expectation-array-item-tag": "",
  "shared.model-trace-explorer.expectation-learn-more-link": "",
  "shared.model-trace-explorer.expectation-value-preview-tooltip": "",
  "shared.model-trace-explorer.feedback-error-item": "",
  "shared.model-trace-explorer.feedback-error-item-stack-trace-link": "",
  "shared.model-trace-explorer.feedback-error-stack-trace-modal": "",
  "shared.model-trace-explorer.feedback-history-modal": "",
  "shared.model-trace-explorer.feedback-learn-more-link": "",
  "shared.model-trace-explorer.feedback-source-count": "",
  "shared.model-trace-explorer.feedback-source-tooltip": "",
  "shared.model-trace-explorer.function-name-tag": "",
  "shared.model-trace-explorer.gateway-trace-link": "",
  "shared.model-trace-explorer.header-details.cost.tag": "",
  "shared.model-trace-explorer.header-details.tag": "",
  "shared.model-trace-explorer.header-details.tooltip": "",
  "shared.model-trace-explorer.header-metadata-pill": "",
  "shared.model-trace-explorer.hide-timeline-info-tooltip": "",
  "shared.model-trace-explorer.image-preview": "",
  "shared.model-trace-explorer.key-value-tag": "",
  "shared.model-trace-explorer.key-value-tag.hover-tooltip": "",
  "shared.model-trace-explorer.key-value-tag.link": "",
  "shared.model-trace-explorer.linked_prompts.table.header": "",
  "shared.model-trace-explorer.linked_prompts.table.search": "",
  "shared.model-trace-explorer.next-search-match": "",
  "shared.model-trace-explorer.prev-search-match": "",
  "shared.model-trace-explorer.relevance-assessment-tooltip": "",
  "shared.model-trace-explorer.retriever-document-collapse": "",
  "shared.model-trace-explorer.right-pane-tabs": "",
  "shared.model-trace-explorer.search-input": "",
  "shared.model-trace-explorer.show-exceptions-tooltip": "",
  "shared.model-trace-explorer.show-parents-tooltip": "",
  "shared.model-trace-explorer.show-timeline-info-tooltip": "",
  "shared.model-trace-explorer.snippet-render-mode-radio": "",
  "shared.model-trace-explorer.snippet-render-mode-tag": "",
  "shared.model-trace-explorer.span-cost-badge": "",
  "shared.model-trace-explorer.span-cost-hovercard.input-cost.tag": "",
  "shared.model-trace-explorer.span-cost-hovercard.output-cost.tag": "",
  "shared.model-trace-explorer.span-cost-hovercard.total-cost.tag": "",
  "shared.model-trace-explorer.span-model-badge": "",
  "shared.model-trace-explorer.span-name-tag": "",
  "shared.model-trace-explorer.span-name-tooltip": "",
  "shared.model-trace-explorer.summary-view.collapse-intermediate-nodes": "",
  "shared.model-trace-explorer.summary-view.expand-intermediate-nodes": "",
  "shared.model-trace-explorer.summary-view.render-mode": "",
  "shared.model-trace-explorer.tag-count": "",
  "shared.model-trace-explorer.tag-count.hover-tooltip": "",
  "shared.model-trace-explorer.text-field-see-more-link": "",
  "shared.model-trace-explorer.timeline-tree-filter-button": "",
  "shared.model-trace-explorer.timeline-tree-filter-popover": "",
  "shared.model-trace-explorer.timeline-tree-node-tooltip": "",
  "shared.model-trace-explorer.timeline-tree-title-time-pill": "",
  "shared.model-trace-explorer.toggle-assessment-expanded": "",
  "shared.model-trace-explorer.toggle-expectation-expanded": "",
  "shared.model-trace-explorer.toggle-graph-button": "",
  "shared.model-trace-explorer.toggle-issue-expanded": "",
  "shared.model-trace-explorer.toggle-show-timeline": "",
  "shared.model-trace-explorer.toggle-span": "",
  "shared.model-trace-explorer.toggle-span-filter": "",
  "shared.model-trace-explorer.toggle-timeline-span": "",
  "shared.model-trace-explorer.token-usage-hovercard.cache-creation-tokens.tag": "",
  "shared.model-trace-explorer.token-usage-hovercard.cached-input-tokens.tag": "",
  "shared.model-trace-explorer.token-usage-hovercard.input-tokens.tag": "",
  "shared.model-trace-explorer.token-usage-hovercard.output-tokens.tag": "",
  "shared.model-trace-explorer.token-usage-hovercard.total-tokens.tag": "",
  "shared.model-trace-explorer.tool-call-id-tooltip": "",
  "shared.model-trace-explorer.trace-too-large.documentation-link": "",
  "shared.model-trace-explorer.trace-too-large.force-display-button": "",
  "shared.model-trace-explorer.view-mode-toggle": "",
  "shared.model-trace-explorer.workflow-node-tooltip": "",
};
