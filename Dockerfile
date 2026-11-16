FROM vllm/vllm-omni:v0.32.0-x86_64@sha256:6b44ee504a0dc3a21eacf72e7dc5da318b7bcc8ed0e349dc928e67278414430c

COPY patches /opt/qwen3-tts-fast/patches
COPY apply-patches.sh deploy_config.py serve.sh register_voice.py /opt/qwen3-tts-fast/
COPY bench /opt/qwen3-tts-fast/bench
RUN bash /opt/qwen3-tts-fast/apply-patches.sh

EXPOSE 8091
ENV HOST=0.0.0.0
ENTRYPOINT ["bash", "/opt/qwen3-tts-fast/serve.sh"]
