import streamlit as st
import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageFont, ImageDraw, ImageOps
import unicodedata
from matplotlib.mlab import specgram
from scipy import fftpack
import soundfile as sf
import tempfile
import os
import base64

def text2specgram(text, start_freq, end_freq, freq_step, step_scale, char_speed, fs, text_inv):
    # 文字数をカウントする
    count = 0
    for n in text:
        if unicodedata.east_asian_width(n) in 'FWA':
            count += 2
        else:
            count += 1

    # 文字列を画像化する 
    image_length = int(count * freq_step / 2 + freq_step)

    img = Image.new(mode='RGB', size=(image_length, freq_step), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    #なんとなくフォント選択（下のは日本語）
    # kbd = ImageFont.truetype("任意のフォント", freq_step)
    kbd = ImageFont.truetype("ipag.ttf", freq_step)
    draw.text((int(freq_step / 2), 0), text, font=kbd, fill=(255, 255, 255))
    img = img.convert("L")  # グレースケール変換

    if text_inv == False:
        img = ImageOps.invert(img)

    # 音声への変換
    image_length_fit = int(image_length / freq_step * char_speed * fs)

    im = np.array(img.resize((image_length_fit, freq_step)))

    wav1 = np.zeros(len(im.T))
    wav = np.zeros(len(im.T))

    for m in range(freq_step):
        if step_scale == True:
            freq = start_freq * (end_freq / start_freq) ** (m / freq_step)
        else:
            freq = start_freq + (end_freq - start_freq) * (m / freq_step)

        for n in range(len(im.T)):
            wav1[n] = np.sin(2 * np.pi * freq * n / fs + m)
        wav = wav + im[freq_step - m - 1, :] * wav1
        wav = wav + wav1

    wav = wav / max(wav) * 24000
    wav = [int(x) for x in wav]

    s_length = len(wav) / fs
    x = np.arange(0, s_length, 1 / fs)

    return img, wav, x

def plot_spectrogram(wav, fs):
    fig, ax = plt.subplots()
    Pxx, freqs, bins, im = ax.specgram(wav, Fs=fs, NFFT=1024, cmap='viridis', noverlap=512)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram')

    return fig

def hide_data_spectrum(original_data, data_to_hide, strength=0.02):
    if original_data.shape[0] < data_to_hide.shape[0]:
        raise ValueError("Data to hide is larger than original data")

    if len(original_data.shape) == 2:
        original_data_left, original_data_right = original_data[:, 0], original_data[:, 1]
        data_to_hide_left, data_to_hide_right = data_to_hide[:, 0], data_to_hide[:, 1]
    else:
        original_data_left, original_data_right = original_data, None
        data_to_hide_left, data_to_hide_right = data_to_hide, None

    padding_left = np.zeros(original_data_left.shape[0] - data_to_hide_left.shape[0])

    data_to_hide_padded_left = np.concatenate((data_to_hide_left, padding_left))
    hidden_freqs_left = fftpack.fft(original_data_left) + (strength * fftpack.fft(data_to_hide_padded_left))
    hidden_data_left = fftpack.ifft(hidden_freqs_left).real

    if original_data_right is not None:
        padding_right = np.zeros(original_data_right.shape[0] - data_to_hide_right.shape[0])
        data_to_hide_padded_right = np.concatenate((data_to_hide_right, padding_right))
        hidden_freqs_right = fftpack.fft(original_data_right) + (strength * fftpack.fft(data_to_hide_padded_right))
        hidden_data_right = fftpack.ifft(hidden_freqs_right).real

        hidden_data = np.stack((hidden_data_left, hidden_data_right), axis=-1)
    else:
        hidden_data = hidden_data_left

    return hidden_data

def remove_silence(data, threshold=0.01):
    for i in range(data.shape[0]-1, 0, -1):
        if np.abs(data[i]).max() > threshold:
            return data[:i+1]
    return data[:1]  

def recover_data_spectrum(hidden_data, original_data, strength=0.02):
    hidden_freqs = fftpack.fft(hidden_data)
    original_freqs = fftpack.fft(original_data)
    recovered_freqs = (hidden_freqs - original_freqs) / strength
    recovered_data = fftpack.ifft(recovered_freqs).real

    recovered_data_resized = np.resize(recovered_data, original_data.shape)
    recovered_data_resized = remove_silence(recovered_data_resized)

    return recovered_data_resized

def get_file_download_link(file_path, download_name):
    with open(file_path, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{download_name}">Download {download_name}</a>'
    return href

def get_file_download_link_spec(data, download_name):
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{download_name}">Download {download_name}</a>'
    return href

    
def main():
    st.title('音声ステガノグラフィー')
    selected_function = st.radio('機能選択:', ('Text to Spectrogram', 'Hide Audio', 'Recover Audio', 'WAV to Spectrogram'))

    if selected_function == 'Text to Spectrogram':
        st.subheader('Text to Spectrogram Converter')

        text = st.text_input('Enter the text', 'Hello world')
        start_freq = st.number_input('Enter start frequency', value=400.0)
        end_freq = st.number_input('Enter end frequency', value=8000.0)
        freq_step = st.number_input('Enter frequency step', value=100)
        step_scale = st.selectbox('Select step scale', [True, False])
        char_speed = st.number_input('Enter character speed', value=1.0)
        # fs = st.number_input('Enter fs', value=44100)
        fs = st.number_input('Enter fs', value=48000)
        text_inv = st.selectbox('Select text inversion', [True, False])

        if st.button('Generate'):
            img, wav, x = text2specgram(text, start_freq, end_freq, freq_step, step_scale, char_speed, fs, text_inv)

            st.image(img)
            fig, ax = plt.subplots()
            ax.plot(x, wav, linewidth=0.5)
            st.pyplot(fig)
            spec_fig = plot_spectrogram(wav, fs)
            st.pyplot(spec_fig)

            buffer = io.BytesIO()
            wavfile = wave.open(buffer, 'w')
            wavfile.setparams((2, 2, fs, len(wav), 'NONE', 'not compressed'))  # 2チャンネルをステレオ用に設定
            wavfile.writeframes(struct.pack("h" * len(wav) * 2, *(wav + wav)))  # 2チャンネル用にデータを2倍に

            wavfile.close()
            buffer.seek(0)
            st.download_button('Download WAV file', buffer, 'text2specgram.wav')

    elif selected_function == 'Hide Audio':
        st.subheader('Hide Audio')

        st.markdown('2つの音声ファイルをアップロードしてください。一つ目には隠し先のファイル、二つ目は隠すファイルです。')
        orig_file = st.file_uploader('Upload Original Audio File', type=['wav'])
        hide_file = st.file_uploader('Upload Audio File to Hide', type=['wav'])
        strength = st.slider('Strength of the Transformatio(変換の強さ)', min_value=0.01, max_value=0.10, value=0.02)

        if orig_file is not None and hide_file is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                orig_file_path = os.path.join(temp_dir, 'orig.wav')
                hide_file_path = os.path.join(temp_dir, 'hide.wav')
                with open(orig_file_path, 'wb') as f:
                    f.write(orig_file.getbuffer())
                with open(hide_file_path, 'wb') as f:
                    f.write(hide_file.getbuffer())

                original_data, original_rate = sf.read(orig_file_path)
                data_to_hide, hide_rate = sf.read(hide_file_path)

                if original_rate != hide_rate:
                    st.error('The two audio files must have the same sample rate.')
                else:
                    hidden_data = hide_data_spectrum(original_data, data_to_hide, strength=strength)
                    hidden_file_path = os.path.join(temp_dir, 'hidden.wav')
                    sf.write(hidden_file_path, hidden_data, original_rate)
                    st.audio(hidden_file_path)

                    # Provide a download link for the hidden audio file
                    st.markdown(get_file_download_link(hidden_file_path, 'hidden.wav'), unsafe_allow_html=True)

    elif selected_function == 'Recover Audio':
        st.subheader('Recover Audio')

        st.markdown('2つの音声ファイルをアップロードしてください。一つ目には隠された音声ファイル、二つ目は元の隠し先ファイルです。')
        hidden_file = st.file_uploader('Upload Hidden Audio File', type=['wav'])
        orig_file = st.file_uploader('Upload Original Audio File', type=['wav'])
        strength = st.slider('Strength of the Transformatio(変換の強さ)', min_value=0.01, max_value=0.10, value=0.02)

        if hidden_file is not None and orig_file is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                hidden_file_path = os.path.join(temp_dir, 'hidden.wav')
                orig_file_path = os.path.join(temp_dir, 'orig.wav')
                with open(hidden_file_path, 'wb') as f:
                    f.write(hidden_file.getbuffer())
                with open(orig_file_path, 'wb') as f:
                    f.write(orig_file.getbuffer())

                hidden_data, hidden_rate = sf.read(hidden_file_path)
                original_data, original_rate = sf.read(orig_file_path)

                if hidden_rate != original_rate:
                    st.error('The two audio files must have the same sample rate.')
                # else:
                #     recovered_data = recover_data_spectrum(hidden_data, original_data, strength=strength)
                #     recovered_file_path = os.path.join(temp_dir, 'recovered.wav')
                #     sf.write(recovered_file_path, recovered_data, original_rate)
                #     st.audio(recovered_file_path)

                #     # Provide a download link for the recovered audio file
                #     st.markdown(get_file_download_link(recovered_file_path, 'recovered.wav'), unsafe_allow_html=True)

                else:
                    recovered_data = recover_data_spectrum(hidden_data, original_data, strength=strength)
                    
                    # リカバーされた音声データを半分にカットする（1chを分析のために無理やり2chに変更しているため）
                    half_length = len(recovered_data) // 2
                    recovered_data = recovered_data[:half_length]

                    recovered_file_path = os.path.join(temp_dir, 'recovered.wav')
                    sf.write(recovered_file_path, recovered_data, original_rate)
                    st.audio(recovered_file_path)

                    st.markdown(get_file_download_link(recovered_file_path, 'recovered.wav'), unsafe_allow_html=True)


    elif selected_function == 'WAV to Spectrogram':
        st.subheader('WAV to Spectrogram Converter')

        # WAV to Spectrogram code
        # WAVファイルのアップロード
        wav_file = st.file_uploader('Choose a WAV file', type=["wav"])

        if wav_file is not None:
            with wav_file as file:
                wav = wave.open(io.BytesIO(file.read()))
                fs = wav.getframerate()
                data = wav.readframes(wav.getnframes())
                wav_data = np.frombuffer(data, dtype=np.int16)

            # サンプリング周波数とサンプル数の表示
            st.write('Sampling Frequency (fs):', fs)
            st.write('Number of samples:', len(wav_data))

            # 波形のプロット
            fig, ax = plt.subplots()
            time = np.linspace(0, len(wav_data) / fs, num=len(wav_data))
            ax.plot(time, wav_data, linewidth=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            st.pyplot(fig)

            # スペクトログラムのプロット
            spec_fig = plot_spectrogram(wav_data, fs)
            st.pyplot(spec_fig)

            # # スペクトログラム画像のダウンロードリンク
            # spec_buffer = io.BytesIO()
            # plt.figure(figsize=(8, 6))
            # plt.specgram(wav_data, Fs=fs, cmap='viridis', NFFT=1024, noverlap=512)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Frequency (Hz)')
            # plt.title('Spectrogram')
            # plt.colorbar(label='Intensity (dB)')
            # plt.tight_layout()
            # plt.savefig(spec_buffer, format='png')
            # st.markdown(get_file_download_link(spec_buffer, 'spectrogram.png'), unsafe_allow_html=True)
            # plt.close()

            # スペクトログラム画像のダウンロードリンク
            spec_buffer = io.BytesIO()
            plt.figure(figsize=(8, 6))
            plt.specgram(wav_data, Fs=fs, cmap='viridis', NFFT=1024, noverlap=512)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectrogram')
            plt.colorbar(label='Intensity (dB)')
            plt.tight_layout()
            plt.savefig(spec_buffer, format='png')
            st.markdown(get_file_download_link_spec(spec_buffer.getvalue(), 'spectrogram.png'), unsafe_allow_html=True)
            plt.close()


if __name__ == '__main__':
    main()

