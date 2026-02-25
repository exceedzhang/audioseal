# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AudioSeal Watermarking Example

This script demonstrates how to:
1. Load an audio file and resample to 16kHz
2. Detect if the audio contains a watermark
3. Add a custom watermark message to the audio
4. Save the watermarked audio as WAV
5. Verify the watermark was added successfully
"""

from typing import Tuple
import time

import torch
import torchaudio

# Disable torchinductor to avoid compilation errors
torch._dynamo.config.suppress_errors = True

from audioseal import AudioSeal
from audioseal.models import AudioSealDetector, AudioSealWM


def load_and_resample_audio(
    file_path: str,
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """Load audio file and resample to target sample rate.

    Args:
        file_path: Path to the audio file (WAV format).
        target_sr: Target sample rate in Hz. Default is 16000.

    Returns:
        A tuple of (audio_tensor, sample_rate).
        The audio tensor has shape (batch, channels, samples).
    """
    wav, sr = torchaudio.load(file_path)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr
        print(f"Resampled from {sr}Hz to {target_sr}Hz")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        print("Converted stereo to mono")

    wav = wav.unsqueeze(0)

    return wav, sr


def detect_watermark(
    audio: torch.Tensor,
    sample_rate: int,
    detector: AudioSealDetector,
) -> Tuple[bool, float, float]:
    """Detect if the audio contains a valid watermark.

    Args:
        audio: Audio tensor with shape (batch, channels, samples).
        sample_rate: Sample rate of the audio in Hz.
        detector: AudioSeal detector model.

    Returns:
        A tuple of (is_valid, probability, time_taken).
        - is_valid: True if watermark detected (probability > 0.5)
        - probability: Detection probability value.
        - time_taken: Time taken in seconds.
    """
    start_time = time.time()
    detector.eval()
    with torch.no_grad():
        result, message = detector.detect_watermark(audio, sample_rate=sample_rate)
    time_taken = time.time() - start_time

    prob = result.mean().item()
    is_valid = prob > 0.5
    return is_valid, prob, time_taken


def string_to_message(text: str, batch_size: int = 1) -> torch.Tensor:
    """Convert a string to a 16-bit binary message tensor.

    Args:
        text: The string message to encode (e.g., "exceed").
        batch_size: Number of batches.

    Returns:
        A tensor of shape (batch_size, 16) with binary values (0 or 1).
    """
    msg_bytes = text.encode("utf-8")
    binary = []
    for b in msg_bytes[:2]:
        for i in range(8):
            binary.append((b >> (7 - i)) & 1)

    while len(binary) < 16:
        binary.append(0)

    binary = binary[:16]

    msg_tensor = torch.tensor([binary], dtype=torch.int32)
    msg_tensor = msg_tensor.repeat(batch_size, 1)
    return msg_tensor


def add_watermark_and_save(
    audio: torch.Tensor,
    sample_rate: int,
    message: str,
    output_path: str,
    generator: AudioSealWM,
) -> float:
    """Add watermark to audio and save as WAV.

    Args:
        audio: Audio tensor with shape (batch, channels, samples).
        sample_rate: Sample rate of the audio in Hz.
        message: Custom message string to embed (e.g., "exceed").
        output_path: Output file path for the watermarked audio.
        generator: AudioSeal generator model.

    Returns:
        Time taken in seconds.
    """
    start_time = time.time()
    msg_tensor = string_to_message(message, batch_size=audio.shape[0])

    generator.eval()
    with torch.no_grad():
        watermark = generator.get_watermark(
            audio, sample_rate=sample_rate, message=msg_tensor
        )
        watermarked_audio = audio + watermark

    if watermarked_audio.shape[0] == 1:
        audio_to_save = watermarked_audio.squeeze(0)
    else:
        audio_to_save = watermarked_audio[0]

    torchaudio.save(output_path, audio_to_save, sample_rate)
    time_taken = time.time() - start_time
    print(f"Saved watermarked audio to: {output_path}")
    return time_taken


def main():
    """Main function to demonstrate watermarking workflow."""
    import argparse

    parser = argparse.ArgumentParser(description="AudioSeal Watermarking Example")
    parser.add_argument("input_file", help="Input WAV file path")
    parser.add_argument(
        "-o", "--output", default="output_watermarked.wav", help="Output WAV file path"
    )
    parser.add_argument(
        "-m", "--message", default="exceed", help="Watermark message to embed"
    )
    parser.add_argument(
        "--detect-only", action="store_true", help="Only detect watermark, do not add"
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output
    watermark_message = args.message
    detect_only = args.detect_only

    if detect_only:
        print("=" * 50)
        print("AudioSeal Watermark Detection")
        print("=" * 50)

        print("\n[1/3] Loading detector model...")
        detector = AudioSeal.load_detector("audioseal_detector_16bits")
        print("Model loaded successfully.")

        print("\n[2/3] Loading and processing audio...")
        audio, sr = load_and_resample_audio(input_file)
        print(f"Audio shape: {audio.shape}, Sample rate: {sr}Hz")

        print("\n[3/3] Detecting watermark...")
        is_valid, prob, detect_time = detect_watermark(audio, sr, detector)
        print(f"  Watermark detected: {is_valid}")
        print(f"  Probability: {prob:.4f}")
        print(f"  Time taken: {detect_time:.4f} seconds")

        print("\n" + "=" * 50)
        print("Done!")
        print("=" * 50)
        return

    print("=" * 50)
    print("AudioSeal Watermarking Example")
    print("=" * 50)

    print("\n[1/5] Loading models...")
    generator = AudioSeal.load_generator("audioseal_wm_16bits")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    print("Models loaded successfully.")

    print("\n[2/5] Loading and resampling audio...")
    audio, sr = load_and_resample_audio(input_file)
    print(f"Audio shape: {audio.shape}, Sample rate: {sr}Hz")

    print("\n[3/5] Detecting watermark in original audio...")
    is_valid, prob, detect_time = detect_watermark(audio, sr, detector)
    print(f"  Watermark detected: {is_valid}")
    print(f"  Probability: {prob:.4f}")
    print(f"  Time taken: {detect_time:.4f} seconds")

    if is_valid:
        print("\n[4/5] Input audio already has watermark, skipping watermark addition.")
        print("\n[5/5] Verification skipped.")
    else:
        print("\n[4/5] Adding watermark to audio...")
        print(f"  Message: '{watermark_message}'")
        add_time = add_watermark_and_save(
            audio, sr, watermark_message, output_file, generator
        )
        print(f"  Time taken: {add_time:.4f} seconds")

        print("\n[5/5] Verifying watermarked audio...")
        audio_wm, sr_wm = load_and_resample_audio(output_file)
        is_valid, prob, verify_time = detect_watermark(audio_wm, sr_wm, detector)
        print(f"  Watermark detected: {is_valid}")
        print(f"  Probability: {prob:.4f}")
        print(f"  Time taken: {verify_time:.4f} seconds")

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
