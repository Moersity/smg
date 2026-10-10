//! `Parallelism::Inline` creates no threads: preprocessing costs the calling
//! thread and nothing else, which is what a host that already runs requests
//! concurrently wants. One process per mode, since the choice is per process.

use image::{DynamicImage, Rgb, RgbImage};
use llm_multimodal::{
    configure_parallelism, vision::processors::Qwen2VLProcessor, Parallelism, PreProcessorConfig,
    VisionPreProcessor,
};

#[cfg(target_os = "linux")]
fn thread_names() -> Vec<String> {
    std::fs::read_dir("/proc/self/task")
        .expect("read Linux thread directory")
        .map(|task| {
            let task = task.expect("read thread entry");
            std::fs::read_to_string(task.path().join("comm")).expect("read thread name")
        })
        .map(|name| name.trim().to_string())
        .collect()
}

fn large_image() -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_pixel(1792, 1344, Rgb([90, 160, 30])))
}

#[test]
fn inline_mode_preprocesses_without_creating_threads() {
    assert_eq!(
        configure_parallelism(Parallelism::Inline),
        Ok(()),
        "this test must own the process's mode"
    );
    #[cfg(target_os = "linux")]
    let before = thread_names();
    let processor = Qwen2VLProcessor::new();
    let config = PreProcessorConfig::default();
    let out = processor
        .preprocess(&[large_image()], &config)
        .expect("preprocess");
    assert!(out.total_feature_tokens() > 0);
    // procfs can observe thread creation on Linux. Other platforms still
    // exercise preprocessing in inline mode above.
    #[cfg(target_os = "linux")]
    {
        use llm_multimodal::vision::execution::POOL_THREAD_NAME_PREFIX;

        let after = thread_names();
        assert!(
            !after
                .iter()
                .any(|name| name.starts_with(POOL_THREAD_NAME_PREFIX)),
            "inline mode spawned pool threads: {after:?}"
        );
        assert_eq!(
            after.len(),
            before.len(),
            "inline mode changed the thread count"
        );
    }
}
