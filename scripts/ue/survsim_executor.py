"""Movie Render Queue executor for SurvSim — deploy to <project>/Content/Python/.

Copy this to `~/data/ue-projects/SurvSim/Content/Python/init_unreal.py`. The
engine auto-runs `init_unreal.py`, and anything defined there becomes
addressable as `/Engine/PythonTypes.<ClassName>`, which is how
`-ExecutorPythonClass` resolves it.

    xvfb-run -a UnrealEditor SurvSim.uproject TestMap -game \
      -MoviePipelineLocalExecutorClass=/Script/MovieRenderPipelineCore.MoviePipelinePythonHostExecutor \
      -ExecutorPythonClass=/Engine/PythonTypes.SurvSimExecutor \
      -LevelSequence=/Game/SurvSim/TestSeq.TestSeq \
      -SurvSimOutput=/tmp/survsim_render -windowed -resx=1280 -resy=720 \
      -unattended -nosplash -nosound

The engine's own example executor exists and was tried first; it renders pure
black here. The differences that matter are the anti-aliasing warm-up and the
game-override setting below — without them the deferred pass is sampled before
lighting and streaming have converged.

**Verify the pixels, never the exit code.** This pipeline reports
`success=True` and writes correctly-named, correctly-sized PNGs that are
entirely black. `tools/check_render.py` exists because "50 frames written" and
"50 frames rendered" are different claims.
"""
WARM_UP_FRAMES = 32
TEMPORAL_SAMPLES = 1
SPATIAL_SAMPLES = 1


@unreal.uclass()
class SurvSimExecutor(unreal.MoviePipelinePythonHostExecutor):
    """Render one Level Sequence to PNG, headless.

    The shipped example executor renders pure black here. The difference is the
    settings below: without a game override and an engine warm-up, the deferred
    pass is sampled before lighting, streaming and the sky have converged, and
    every frame comes out empty while the render still reports success.
    """

    pipeline = unreal.uproperty(unreal.MoviePipeline)
    queue = unreal.uproperty(unreal.MoviePipelineQueue)
    seq_path = unreal.uproperty(unreal.Text)

    def _post_init(self):
        self.pipeline = None
        self.queue = None
        self.seq_path = ""

    @unreal.ufunction(override=True)
    def execute_delayed(self, in_queue):
        (_tokens, _switches, params) = unreal.SystemLibrary.parse_command_line(
            unreal.SystemLibrary.get_command_line())
        sequence = params.get("LevelSequence")
        output_dir = params.get("SurvSimOutput", "/tmp/survsim_render")
        if not sequence:
            unreal.log_error("[survsim] missing -LevelSequence=")
            self.on_executor_errored()
            return

        unreal.log(f"[survsim] rendering {sequence} -> {output_dir}")
        self.queue = unreal.new_object(unreal.MoviePipelineQueue, outer=self)
        job = self.queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.sequence = unreal.SoftObjectPath(sequence)
        job.map = unreal.SoftObjectPath(params.get("SurvSimMap", ""))

        config = job.get_configuration()
        config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)

        out = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        out.output_directory = unreal.DirectoryPath(output_dir)
        out.output_resolution = unreal.IntPoint(1280, 720)
        out.file_name_format = "{sequence_name}.{frame_number}"
        out.override_existing_output = True

        # The fix for black frames: let the renderer converge before sampling.
        aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
        aa.spatial_sample_count = SPATIAL_SAMPLES
        aa.temporal_sample_count = TEMPORAL_SAMPLES
        aa.engine_warm_up_count = WARM_UP_FRAMES
        aa.render_warm_up_frames = True

        # Forces cinematic quality and disables the game-mode behaviours that
        # suppress rendering in a headless run.
        config.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
        config.initialize_transient_settings()

        self.pipeline = unreal.new_object(
            self.target_pipeline_class, outer=self.get_last_loaded_world(),
            base_type=unreal.MoviePipeline)
        self.pipeline.on_movie_pipeline_work_finished_delegate.add_function_unique(
            self, "on_job_finished")
        self.pipeline.initialize(job)

    @unreal.ufunction(override=True)
    def is_rendering(self):
        return False

    @unreal.ufunction(ret=None, params=[unreal.MoviePipelineOutputData])
    def on_job_finished(self, results):
        unreal.log(f"[survsim] render finished, success={results.success}")
        self.pipeline = None
        self.on_executor_finished_impl()
