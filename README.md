<div align="center">
	<!-- Welcome words -->
	<h1 style="color: #FFA500;font-size: 36px;">
	<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcnA3cDMyenY4OWpkZHU5OGJ4anJpdGJkZG02eWlmcGVhdHVzY215aSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jIWdDlz1s5a7k1o0w0/giphy.gif" alt="yf" width="70" height="50" />
  🎧 Welcome to AudioCodecBench 🎵
	<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcnA3cDMyenY4OWpkZHU5OGJ4anJpdGJkZG02eWlmcGVhdHVzY215aSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jIWdDlz1s5a7k1o0w0/giphy.gif" alt="yf" width="70" height="50" />
 	</h1>
	<!-- Dynamic Emojis -->
	<div style="display: flex; justify-content: center; align-items: center;">
	<img src="https://github.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/blob/master/Emojis/Objects/Musical%20Notes.png" alt="Waving Hand" width="50" height="50" />
  	<img src="https://github.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/blob/master/Emojis/Objects/Guitar.png" alt="Waving Hand" width="50" height="50" />
	<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Hand%20gestures/Waving%20Hand.png" alt="Waving Hand" width="50" height="50" />
  	<img src="https://github.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/blob/master/Emojis/Objects/Saxophone.png" alt="Glowing Star" width="50" height="50" />
  	<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Glowing%20Star.png" alt="Glowing Star" width="50" height="50" />
	<img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Headphone/3D/headphone_3d.png" alt="Fire" width="50" height="50" />
 	<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Fire.png" alt="Fire" width="50" height="50" />
	<img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Violin/3D/violin_3d.png" alt="Sun" width="50" height="50" />  
	<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Sun.png" alt="Sun" width="50" height="50" />
	<img src="https://github.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/blob/master/Emojis/Objects/Trumpet.png" alt="Sun" width="50" height="50" />
	<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="50" height="50" />
	</div>
</div>


# AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation
&emsp;&emsp;__AudioCodecBench__ evaluates audio codecs in terms of reconstruction quality and the adaptability of generated representations to large language model(LLM) modeling, by performing tasks such as reconstruction (adding noise to assess the codec’s ability to fit noise and to determine whether the codebook information tends to encode noise), audio offset (simulating phase shifts to evaluate noise fitting ability), mutual information (analyzing the correlation between codebook tokens before and after reconstruction to assess the stability of information representation), and downstream probe tasks (simulating large language model-style modeling).<br>
 <a href="https://www.overleaf.com/project/67e23cf832acf0f4e121f472">
  <img src="https://camo.githubusercontent.com/a8d50b4cb0bebfa879fca60626080da8012c5a24a1fe3d3db641e19485b0851f/68747470733a2f2f7374617469632e61727869762e6f72672f7374617469632f62726f7773652f302e332e342f696d616765732f69636f6e732f66617669636f6e2d31367831362e706e67" alt="arXiv Paper: AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation">
  arXiv Paper: AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation
</a>

<div style="display: flex; justify-content: center; align-items: left;">
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Dove.png" alt="Dove" width="20" height="20">contribution1 <br>
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Peacock.png" alt="Peacock" width="20" height="20" />contribution2 <br>
<img src="https://github.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/blob/master/Emojis/Animals/Unicorn.png" alt="unicorn" width="20" height="20" />contribution3 <br>
<img src="https://github.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/blob/master/Emojis/Animals/Fox.png" alt="fox" width="20" height="20" />contribution4 <br>
<img src="https://github.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/blob/master/Emojis/Animals/Penguin.png" alt="penguin" width="20" height="20" />contribution5 <br>
</div>

## <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Mountain/3D/mountain_3d.png" alt="mountain" width="30" height="30"> Purpose

1. how to evaluate the quality of codebook
2. collect all existing metrics for reconstruction
3. collect all existing metrics for Linear Probing (Music and Speech)

## <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/World%20map/3D/world_map_3d.png" alt="map" width="30" height="30"> Road Map

- [x] multi codec deploy
    - multi codec deploy reference: https://github.com/lucadellalib/audiocodecs
- [ ] clean different dataset in marble benchmark
    - add code to redeploy marble in our benchmark
    - add code in marble base to evaluate our index 
- [ ] define the evaluation metrics of codec, codebooks
    - test ID sensitive in same semantic
    - 

## <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Compass/3D/compass_3d.png" alt="compass" width="30" height="30"> Env Build
The following explains how to quickly create the required environment and install codec_evaluation for use.

### Setup environment and dependencies
We strongly recommended to use conda for managing your Python environment.

- #### Create a virtual environment using conda.
	```
	# create a virtual environment using conda.
	conda create -n codec_eval python==3.10 -y	# Python ==3.10 is recommended.
	conda activate codec_eval
	```
- #### Install codec_evaluation from source
  	```
 	git clone https://github.com/wuzhiyue111/Codec-Evaluation.git
	cd Codec-Evaluation
	bash env_build.sh
   	```
## <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Straight%20ruler/3D/straight_ruler_3d.png" alt="ruler" width="30" height="30"> Usage
The following will introduce how to conduct evaluations using codecs and downstream tasks. 

### Audio codec
Currently, 10 codecs have been added to the repository. Each codec program has five modes, namely `encode`, `decode`, `reconstruct`, `unquantized_emb`, and `unquantized_emb`. In the code, we've designed two ways to load model weights: local and remote. Users can choose either loading method according to their own needs. 

<table>
  <tr>
    <th>Codec</th>
    <th>DAC</th>
    <th>Encodec</th>
    <th>Mimi</th>
    <th>WavTokenizer</th>
    <th>Semanticodec</th>
    <th>Speechtokenizer</th>
    <th>Qwen2AudioEncoder</th>
    <th>HuBert</th>
    <th>XCodec</th>
    <th>YuE</th>
  </tr>
  <tr>
    <td>Sample-rate</td>
    <td>24kHz</td>
    <td>24kHz</td>
    <td>24kHz</td>
    <td>24kHz</td>
    <td>16kHz</td>
    <td>16kHz</td>
    <td>16kHz</td>
    <td>16kHz</td>
    <td>16kHz</td>
    <td>16kHz</td>
  </tr>
  <tr>
    <td>Modes</td>
    <td colspan="10">
	encode: encode the audio to id tokens <br>
        decode: decode the id tokens to audio <br>
        reconstruct: encode -> decode <br>
        unquantized_emb: encode -> unquantized embedding <br>
        quantized_emb: encode + quantizer -> quantized embedding <br>
    </td>
  </tr>
</table>



## Reconstruction Metric
### Speech
<table border="1" align="center">
  <thead>
    <tr>
      <th rowspan="2">Codec</th>
      <th colspan="8">Metrics</th>
    </tr>
    <tr>
      <th>PESQ</th>
      <th>Speaker_Sim</th>
      <th>WER_GT</th>
      <th>WER_REC</th>
      <th>CER_GT</th>
      <th>CER_REC</th>
      <th>STOI</th>
      <th>SISNR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DAC</td>
      <td>3.69</td>
      <td>0.965</td>
      <td>0.155</td>
      <td>0.202</td>
      <td>0.09</td>
      <td>0.125</td>
      <td>0.94</td>
      <td>0.149</td>
    </tr>
    <tr>
      <td>Encodec</td>
      <td>3.21</td>
      <td>0.919</td>
      <td>0.155</td>
      <td>0.198</td>
      <td>0.09</td>
      <td>0.114</td>
      <td>0.925</td>
      <td>6.902</td>
    </tr>
    <tr>
      <td>Mimi</td>
      <td>2.77</td>
      <td>0.928</td>
      <td>0.155</td>
      <td>0.287</td>
      <td>0.09</td>
      <td>0.173</td>
      <td>0.88</td>
      <td>4.162</td>
    </tr>
    <tr>
      <td>SemantiCodec</td>
      <td>2.64</td>
      <td>0.907</td>
      <td>0.155</td>
      <td>0.318</td>
      <td>0.09</td>
      <td>0.195</td>
      <td>0.86</td>
      <td>-2.484</td>
    </tr>
    <tr>
      <td>WavTokenizer</td>
      <td>2.17</td>
      <td>0.743</td>
      <td>0.155</td>
      <td>0.494</td>
      <td>0.09</td>
      <td>0.325</td>
      <td>0.83</td>
      <td>1.318</td>
    </tr>
    <tr>
      <td>SpeechTokenizer</td>
      <td>2.97</td>
      <td>0.924</td>
      <td>0.155</td>
      <td>0.216</td>
      <td>0.09</td>
      <td>0.12</td>
      <td>0.89</td>
      <td>3.495</td>
    </tr>
    <tr>
      <td>XCodec</td>
      <td>3.23</td>
      <td>0.942</td>
      <td>0.154</td>
      <td>0.185</td>
      <td>0.09</td>
      <td>0.106</td>
      <td>0.91</td>
      <td>-0.493</td>
    </tr>
    <tr>
      <td>YuE</td>
      <td>3.17</td>
      <td>0.938</td>
      <td>0.154</td>
      <td>0.195</td>
      <td>0.09</td>
      <td>0.113</td>
      <td>0.90</td>
      <td>-0.472</td>
    </tr>
  </tbody>
</table>

### Music
<table border="1" align="center">
  <thead>
    <tr>
      <th rowspan="3">Codec</th>
      <th colspan="5">Metrics</th>
    </tr>
    <tr>
      <th>PESQ</th>
      <th>STOI</th>
      <th>SISNR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DAC</td>
      <td>2.66</td>
      <td>0.86</td>
      <td>0.980</td>
    </tr>
    <tr>
      <td>Encodec</td>
      <td>2.27</td>
      <td>0.85</td>
      <td>5.891</td>
    </tr>
    <tr>
      <td>SemantiCodec</td>
      <td>1.32</td>
      <td>0.60</td>
      <td>-3.086</td>
    </tr>
    <tr>
      <td>WavTokenizer</td>
      <td>1.14</td>
      <td>0.49</td>
      <td>-0.464</td>
    </tr>
    <tr>
       <td>XCodec</td>
       <td>1.85</td>
       <td>0.76</td>
       <td>-0.065</td>
    </tr>
    <tr>
       <td>YuE</td>
       <td>1.84</td>
       <td>0.75</td>
       <td>-0.296</td>
    </tr>
  </tbody>
</table>

## Probe Experiment
### Marble Probe (Music)
<table border="1" >
    
   <thead>
        <tr>
            <th rowspan="3">Codec</th>
	    <th rowspan="3">Mode</th>
            <th colspan="19">Dataset</th>
        </tr>
        <tr>
            <th colspan="2">emomusic</th>
            <th colspan="1">GTZAN</th>
            <th colspan="2">MTT</th>
            <th colspan="1">NSynthI</th>
            <th colspan="1">NSynthP</th>
	    <th colspan="1">VocalSetSinger</th>
	    <th colspan="1">VocalSetTech</th>
	    <th colspan="1">GS</th>
	    <th colspan="1">Muchin</th>
	    <th colspan="2">MTGGenre</th>
	    <th colspan="2">MTGInstrument</th>
	    <th colspan="2">MTGMoodtheme</th>
	    <th colspan="2">MTGTop50</th>
        </tr>
	<tr>
	    <td align=center>A</td>
            <td align=center>V</td>
	    <td align=center>Acc</td>
            <td align=center>AP</td>
	    <td align=center>AUCROC</td>
	    <td align=center>Acc</td>
	    <td align=center>Acc</td>
	    <td align=center>Acc</td>
	    <td align=center>Acc</td>
	    <td align=center>Acc</td>	   
	    <td align=center>CER</td>
	    <td align=center>AP</td>
	    <td align=center>AUCROC</td>
	    <td align=center>AP</td>
	    <td align=center>AUCROC</td>
	    <td align=center>AP</td>
	    <td align=center>AUCROC</td>
	    <td align=center>AP</td>
	    <td align=center>AUCROC</td>
    </thead>
    <tbody>
        <tr>
            <td align=center rowspan="2">DAC</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.470</td>
            <td align=center>0.064</td>
            <td align=center>0.353</td>
            <td align=center>0.203</td>
	    <td align=center>0.785</td>
	    <td align=center>0.602</td>
	    <td align=center>0.468</td>
	    <td align=center>0.419</td>
	    <td align=center>0.376</td>
            <td align=center>0.088</td>
	    <td align=center>0.579</td>
	    <td align=center>0.0295</td>
	    <td align=center>0.530</td>
	    <td align=center>0.108</td>
	    <td align=center>0.638</td>
            <td align=center>0.076</td>
	    <td align=center>0.651</td>
	    <td align=center>0.141</td>
	    <td align=center>0.687</td>
        </tr>
        <tr>
            <td align=center rowspan="2">Encodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.467</td>
            <td align=center>0.066</td>
            <td align=center>0.339</td>
            <td align=center>0.184</td>
            <td align=center>0.759</td>
	    <td align=center>0.537</td>
	    <td align=center>0.547</td>
	    <td align=center>0.299</td>
	    <td align=center>0.301</td>
	    <td align=center>0.102</td>
	    <td align=center>0.507</td>
	    <td align=center>0.035</td>
	    <td align=center>0.528</td>
	    <td align=center>0.104</td>
	    <td align=center>0.620</td>
            <td align=center>0.057</td>
	    <td align=center>0.642</td>
	    <td align=center>0.137</td>
	    <td align=center>0.701</td>
        </tr>
        <tr>
            <td align=center rowspan="2">SemantiCodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.507</td>
            <td align=center>0.316</td>
            <td align=center>0.502</td>
            <td align=center>0.318</td>
            <td align=center>0.877</td>
	    <td align=center>0.658</td>
	    <td align=center>0.764</td>
	    <td align=center>0.344</td>
	    <td align=center>0.451</td>
	    <td align=center>0.343</td>
	    <td align=center>0.578</td>
	    <td align=center>0.035</td>
	    <td align=center>0.526</td>
	    <td align=center>0.149</td>
	    <td align=center>0.720</td>
            <td align=center>0.099</td>
	    <td align=center>0.723</td>
	    <td align=center>0.230</td>
	    <td align=center>0.795</td>
        </tr>
        <tr>
            <td align=center rowspan="2">WavTokenizer</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.455</td>
            <td align=center>0.066</td>
            <td align=center>0.329</td>
            <td align=center>0.168</td>
            <td align=center>0.739</td>
	    <td align=center>0.537</td>
	    <td align=center>0.444</td>
	    <td align=center>0.130</td>
	    <td align=center>0.287</td>
	    <td align=center>0.093</td>
            <td align=center>0.721</td>
	    <td align=center>0.034</td>
	    <td align=center>0.530</td>
	    <td align=center>0.107</td>
	    <td align=center>0.635</td>
            <td align=center>0.056</td>
	    <td align=center>0.627</td>
	    <td align=center>0.137</td>
	    <td align=center>0.698</td>
        </tr>
    </tbody>
</table>

### Speech and Sound Probe
<table border="1" >
    
   <thead>
        <tr>
            <th rowspan="3">Codec</th>
	    <th rowspan="3">Mode</th>
            <th colspan="12">Dataset</th>
        </tr>
        <tr>
            <th colspan="2">libritts</th>
            <th colspan="1">Vocalsound</th>
            <th colspan="1">MELD</th>
        </tr>
	<tr>
	    <td align=center>WER</td>
            <td align=center>CER</td>
	    <td align=center>Acc</td>
            <td align=center>Acc</td>
    </thead>
    <tbody>
        <tr>
            <td align=center rowspan="2">DAC</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.573</td>
            <td align=center>0.369</td>
            <td align=center>0.535</td>
            <td align=center>0.483</td>
        </tr>
        <tr>
            <td align=center rowspan="2">Encodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.584</td>
            <td align=center>0.369</td>
	    <td align=center>0.574</td>
	    <td align=center>0.481</td>
        </tr>
        <tr>
            <td align=center rowspan="2">SemantiCodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.445</td>
            <td align=center>0.183</td>
            <td align=center>0.723</td>
            <td align=center>0.482</td>
        </tr>
        <tr>
            <td align=center rowspan="2">WavTokenizer</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.645</td>
            <td align=center>0.416</td>
            <td align=center>0.524</td>
            <td align=center>0.484</td>
        </tr>
	<tr>
	    <td align=center rowspan="2">Mimi</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
	</tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.499</td>
            <td align=center>0.22</td>
            <td align=center>0.833</td>
            <td align=center>0.481</td>
        </tr>
	<tr>
	    <td align=center rowspan="2">SpeechTokenizer</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
            <td align=center></td>
	</tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.433</td>
            <td align=center>0.164</td>
            <td align=center>0.776</td>
            <td align=center>0.498</td>
        </tr>
    </tbody>
</table>






