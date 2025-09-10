<div align="center">
	<!-- Welcome words -->
	<h1 style="color: #FFA500;font-size: 36px;">
<!-- 	<img src="https://media.tenor.com/92w2JUO-fk4AAAAi/musical-notes.gif" alt="yf" width="50" height="50" /> -->
<!--  	<img src="https://media.tenor.com/Ziew1c0_mKUAAAAi/pepe-frog.gif" alt="yf" width="50" height="50" /> -->
  🎧 Welcome to AudioCodecBench 🎵
<!--   	<img src="https://media.tenor.com/2wq1PfInyYkAAAAj/music-note-dancing.gif" alt="yf" width="50" height="50" /> -->
<!-- 	<img src="https://media.tenor.com/92w2JUO-fk4AAAAi/musical-notes.gif" alt="yf" width="50" height="50" /> -->
 	</h1>
	<!-- Dynamic Emojis -->
<!-- 	<div style="display: flex; justify-content: center; align-items: center;">
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
	<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="50" height="50" /> -->
	</div>
</div>


# AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation
&emsp;&emsp;__AudioCodecBench__ allows for a comprehensive assessment of codecs' capabilities which evaluate across four dimensions: audio reconstruction metric, codebook index (ID) stability, decoder-only transformer perplexity, and performance on downstream probe tasks. Our results show the correctness of the provided suitable definitions and the correlation among reconstruction metrics, codebook ID stability, downstream probe tasks and perplexity.<br>
 <a href="https://arxiv.org/pdf/2509.02349">
  <img src="https://camo.githubusercontent.com/a8d50b4cb0bebfa879fca60626080da8012c5a24a1fe3d3db641e19485b0851f/68747470733a2f2f7374617469632e61727869762e6f72672f7374617469632f62726f7773652f302e332e342f696d616765732f69636f6e732f66617669636f6e2d31367831362e706e67" alt="arXiv Paper: AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation">
  arXiv Paper: AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation
</a>

<!-- <div style="display: flex; justify-content: center; align-items: left;">  <br> -->
<!-- <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Dove.png" alt="Dove" width="20" height="20">contribution1 <br> -->
<!-- <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Peacock.png" alt="Peacock" width="20" height="20" />contribution2 <br> -->
<!-- <img src="https://github.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/blob/master/Emojis/Animals/Unicorn.png" alt="unicorn" width="20" height="20" />contribution3 <br> -->
<!-- <img src="https://github.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/blob/master/Emojis/Animals/Fox.png" alt="fox" width="20" height="20" />contribution4 <br> -->
<!-- <img src="https://github.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/blob/master/Emojis/Animals/Penguin.png" alt="penguin" width="20" height="20" />contribution5 <br> -->
<!-- </div> <br> -->

## <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Mountain/3D/mountain_3d.png" alt="mountain" width="30" height="30"> Purpose

1. how to evaluate the quality of codebook
2. collect all existing metrics for reconstruction
3. collect all existing metrics for Linear Probing (Music and Speech)

## <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/World%20map/3D/world_map_3d.png" alt="map" width="30" height="30"> Road Map

- [x] multi codec deploy
    - multi codec deploy reference: https://github.com/lucadellalib/audiocodecs
- [x] clean different dataset in marble benchmark
    - add code to redeploy marble in our benchmark
    - add code in marble base to evaluate our index
    - package default behavior: load ckpt or dataset from default base dir (like: ~/.codec_evaluation) or os environment var (like CODEC_EVALUATION_DATA_DIR) rather than absolute path
- [x] define the evaluation metrics of codec, codebooks
    - test codec's reconstruction fidelity
    - test ID sensitive in same semantic
    - test perplexity(ppl) of the different token sequences on the same LMs
    - test token embeddings in downstream probe tasks

## <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Compass/3D/compass_3d.png" alt="compass" width="30" height="30"> Env Build
The following explains how to quickly create the required environment and install codec_evaluation for use.

### Setup environment and dependencies
We strongly recommended to use conda for managing your Python environment.

- #### Create a virtual environment using conda.
	```
	# create a virtual environment using conda
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
The following will introduce how to conduct evaluations using codecs and downstream tasks. For details, please refer to the instruction document. <a href="https://tcn29bn4ijcy.feishu.cn/wiki/U67rwNej8i2BGWkzwqQcuMGanj7?from=from_copylink">[EN]</a><a href="https://tcn29bn4ijcy.feishu.cn/wiki/ZO0dwidYti4UEgkK7BHcfXE8nnh?from=from_copylink">[ZH]</a>

### <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Toolbox/Flat/toolbox_flat.svg" alt="toolbox" width="30" height="30"> Probe 
  
## <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Pen/3D/pen_3d.png" alt="pen" width="30" height="30">Probe task results
### <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Bookmark%20tabs/3D/bookmark_tabs_3d.png" alt="bookmark" width="30" height="30">Reconstruction Metric
#### Speech
<table border="1">
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
    </tr>
    <tr>
      <td>XCodec</td>
      <td>3.23</td>
      <td>0.942</td>
      <td>0.155</td>
      <td>0.185</td>
      <td>0.09</td>
      <td>0.106</td>
      <td>0.91</td>
    </tr>
    <tr>
      <td>YuE</td>
      <td>3.17</td>
      <td>0.938</td>
      <td>0.155</td>
      <td>0.195</td>
      <td>0.09</td>
      <td>0.113</td>
      <td>0.90</td>
    </tr>
  </tbody>
</table>

#### Music
<table border="1" >
  <thead>
    <tr>
      <th rowspan="3">Codec</th>
      <th colspan="5">Metrics</th>
    </tr>
    <tr>
      <th>PESQ</th>
      <th>STOI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DAC</td>
      <td>2.66</td>
      <td>0.86</td>
    </tr>
    <tr>
      <td>Encodec</td>
      <td>2.27</td>
      <td>0.85</td>
    </tr>
    <tr>
      <td>SemantiCodec</td>
      <td>1.32</td>
      <td>0.60</td>
    </tr>
    <tr>
      <td>WavTokenizer</td>
      <td>1.14</td>
      <td>0.49</td>
    </tr>
    <tr>
       <td>XCodec</td>
       <td>1.85</td>
       <td>0.76</td>
    </tr>
    <tr>
       <td>YuE</td>
       <td>1.84</td>
       <td>0.75</td>
    </tr>
  </tbody>
</table>

### <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/Bookmark%20tabs/3D/bookmark_tabs_3d.png" alt="bookmark" width="30" height="30">Probe Experiment
#### Music Probe
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

#### Speech and Sound Probe
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
            <th colspan="1">ESC50</th>
        </tr>
	<tr>
	    <td align=center>WER</td>
            <td align=center>CER</td>
	    <td align=center>Acc</td>
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
            <td align=center></td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.573</td>
            <td align=center>0.369</td>
            <td align=center>0.535</td>
            <td align=center>0.483</td>
	    <td align=center>0.325</td>
        </tr>
        <tr>
            <td align=center rowspan="2">Encodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
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
	    <td align=center>0.275</td>
        </tr>
        <tr>
            <td align=center rowspan="2">SemantiCodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
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
	    <td align=center>0.620</td>
        </tr>
        <tr>
            <td align=center rowspan="2">WavTokenizer</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
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
	    <td align=center>0.135</td>
        </tr>
	<tr>
	    <td align=center rowspan="2">Mimi</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
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
	    <td align=center>0.335</td>
        </tr>
	<tr>
	    <td align=center rowspan="2">SpeechTokenizer</td>
            <td align=center>unquantized_emb</td>
            <td align=center></td>
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
	    <td align=center>0.670</td>
        </tr>
    </tbody>
</table>

### PPL Experiment
#### LibriTTS

| Codec          | ppl↓   |  cb1_ppl | cb2_ppl | cb3_ppl | cb4_ppl | cb5_ppl | cb6_ppl | cb7_ppl | cb8_ppl |
|----------------|--------|----------|---------|---------|---------|---------|---------|---------|---------|
| DAC            |  420.6 |  48.9    | 284.1   | 428.6   | 560.2   | 609.7   | 728.1   | 814.0   | 835.5   |
| Encodec        |  111.4 |  28.0    | 59.1    | 93.7    | 130.5   | 153.7   | 183.3   | 202.0   | 213.5   |         
| WavTokenizer   |  317.1 |  317.1   |    -    |    -    |    -    |    -    |    -    |    -    |    -    |
| X-Codec        |  56.2  |  20.6    | 24.9    | 37.8    | 57.3    | 77.3    | 92.0    | 103.0   | 126.3   |
| YuE            |  52.7  |  18.3    | 29.6    | 37.7    | 52.6    | 74.3    | 89.8    | 95.3    | 90.3    |
| SpeechTokenizer|  24.2  |  2.4     | 12.4    | 24.8    | 33.6    | 40.9    | 46.0    | 50.3    | 52.8    |
| Mimi           |  269.6 |  32.9    | 189.3   | 334.7   | 383.3   | 424.2   | 431.7   | 456.9   | 459.7   |
| SemamiCodec    |  14.8  |  1.2     | 191.0   |    -    |    -    |    -    |    -    |    -    |    -    |

#### Emilia_EN(100ksteps)

| Codec          | ppl↓   | cb1_ppl  | cb2_ppl | cb3_ppl | cb4_ppl | cb5_ppl | cb6_ppl | cb7_ppl | cb8_ppl |
|----------------|--------|----------|---------|---------|---------|---------|---------|---------|---------|
| DAC            | 247    |  20.6    |  146.7  |  218    |  315.1  |  395.9  |  482.9  |  569.6  |  628.2  |
| Encodec        | 75.7   |  14.8    |  33.4   |  59     |  88.7   |  111.3  |  138.4  |  158.5  |  172.6  |
| WavTokenizer   | 104.7  |  104.7   |    -    |    -    |    -    |    -    |    -    |    -    |    -    |
| X-Codec        | 30.3   |  10.0    |  12.7   |  20.2   |  30.6   |  41.9   |  50.7   |  61.6   |  71.4   |
| YuE            | 29.0   |  9.3     |  16.0   |  19.9   |  29.3   |  38.7   |  51.0   |  55.2   |  54.1   |         
| SpeechTokenizer| 13.5   |  1.9     |  5.5    |  12.1   |  18.3   |  22.3   |  25.1   |  28.6   |  30.8   |
| Mimi           | 126.9  |  9.1     |  58.2   |  148.0  |  185.0  |  228.7  |  256.6  |  278.9  |  298.5  |        
| SemamiCodec    | 7.9    |  1.0     |  82.1   |    -    |    -    |    -    |    -    |    -    |    -    |

#### MTG-Jamendo(100ksteps)

| Codec          | ppl↓  | cb1_ppl | cb2_ppl | cb3_ppl | cb4_ppl | cb5_ppl | cb6_ppl | cb7_ppl | cb8_ppl |
|----------------|-------|---------|---------|---------|---------|---------|---------|---------|---------|
| DAC            | 194   |  28.6   |  122.8  |  152.4  |  212.8  |  270.7  |  352.9  |  413.4  |  473.5  |
| Encodec        | 141.3 |  17.6   |  62.5   |  110.7  |  170    |  225.9  |  287    |  336.8  |  375.6  |
| WavTokenizer   | 38.2  |  38.2   |    -    |    -    |    -    |    -    |    -    |    -    |    -    |
| X-Codec        | 47.5  |  20.4   |  19.6   |  32.4   |  51.1   |  64.5   |  74.5   |  86.8   |  100.2  |
| YuE            | 46.2  |  18.3   |  28.7   |  30.4   |  48.2   |  60.0   |  74.9   |  83.0   |  76.3   |                
| SemamiCodec    | 15.5  |  1.0    |  272.4  |    -    |    -    |    -    |    -    |    -    |    -    |


## Citation
```
@misc{wang2025audiocodecbenchcomprehensivebenchmarkaudio,
      title={AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation}, 
      author={Lu Wang and Hao Chen and Siyu Wu and Zhiyue Wu and Hao Zhou and Chengfeng Zhang and Ting Wang and Haodi Zhang},
      year={2025},
      eprint={2509.02349},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2509.02349}, 
}
```
