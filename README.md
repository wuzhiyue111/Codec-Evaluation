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

1. how to evaluate the quality of codebook (for lm modeling)
2. collect all existing metrics for reconstruction
3. collect all existing metrics for Linear Probing (Music and Speech)

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

## <img src="https://github.com/microsoft/fluentui-emoji/blob/main/assets/World%20map/3D/world_map_3d.png" alt="map" width="30" height="30"> Dataset Download 
Dataset download address: <a href="https://huggingface.co/datasets/LeBeGut/AudioCodecBench">AudioCodecBench-Dataset</a>


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
      <td><strong>3.69<strong/></td>
      <td><strong>0.965<strong/></td>
      <td>0.155</td>
      <td>0.202</td>
      <td>0.09</td>
      <td>0.125</td>
      <td><strong>0.94<strong/></td>
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
      <td><strong>0.185<strong/></td>
      <td>0.09</td>
      <td><strong>0.106<strong/></td>
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
      <td><strong>2.66<strong/></td>
      <td><strong>0.86<strong/></td>
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
            <td align=center>DAC</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.470</td>
            <td align=center>0.064</td>
            <td align=center>0.575</td>
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
            <td align=center>Encodec</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.467</td>
            <td align=center>0.066</td>
            <td align=center>0.570</td>
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
            <td align=center>SemantiCodec</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.507</td>
            <td align=center><strong>0.316<strong/></td>
            <td align=center><strong>0.703<strong/></td>
            <td align=center>0.318</td>
            <td align=center><strong>0.877<strong/></td>
		    <td align=center><strong>0.658<strong/></td>
		    <td align=center>0.764</td>
		    <td align=center>0.344</td>
		    <td align=center>0.451</td>
		    <td align=center>0.343</td>
		    <td align=center>0.578</td>
		    <td align=center><strong>0.035<strong/></td>
		    <td align=center>0.526</td>
		    <td align=center>0.149</td>
		    <td align=center><strong>0.720<strong/></td>
			<td align=center>0.099</td>
		    <td align=center><strong>0.723<strong/></td>
		    <td align=center><strong>0.230<strong/></td>
		    <td align=center><strong>0.795<strong/></td>
        </tr>
        <tr>
            <td align=center>WavTokenizer</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.455</td>
            <td align=center>0.066</td>
            <td align=center>0.423</td>
            <td align=center>0.168</td>
            <td align=center>0.739</td>
		    <td align=center>0.537</td>
		    <td align=center>0.444</td>
		    <td align=center>0.130</td>
		    <td align=center>0.287</td>
		    <td align=center>0.093</td>
            <td align=center><strong>0.721<strong/></td>
		    <td align=center>0.034</td>
		    <td align=center><strong>0.530<strong/></td>
		    <td align=center>0.107</td>
		    <td align=center>0.635</td>
            <td align=center>0.056</td>
		    <td align=center>0.627</td>
		    <td align=center>0.137</td>
		    <td align=center>0.698</td>
        </tr>
	<tr>
            <td align=center>Xcodec</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.553</td>
            <td align=center>0.143</td>
            <td align=center>0.664</td>
            <td align=center><strong>0.323<strong/></td>
		    <td align=center>0.873</td>
		    <td align=center>0.640</td>
		    <td align=center><strong>0.905<strong/></td>
		    <td align=center><strong>0.537<strong/></td>
		    <td align=center>0.570</td>
            <td align=center><strong>0.455<strong/></td>
		    <td align=center>0.481</td>
		    <td align=center>0.034</td>
		    <td align=center>0.519</td>
		    <td align=center><strong>0.164<strong/></td>
		    <td align=center>0.707</td>
            <td align=center>0.101</td>
		    <td align=center>0.710</td>
		    <td align=center>0.216</td>
		    <td align=center>0.777</td>
        </tr>
	<tr>
            <td align=center>YuE</td>
            <td align=center>quantized_emb</td>
            <td align=center><strong>0.573<strong/></td>
            <td align=center>0.156</td>
            <td align=center>0.669</td>
            <td align=center>0.315</td>
		    <td align=center>0.870</td>
		    <td align=center>0.622</td>
		    <td align=center>0.896</td>
		    <td align=center>0.523</td>
		    <td align=center><strong>0.594<strong/></td>
            <td align=center>0.454</td>
		    <td align=center>0.463</td>
		    <td align=center>0.034</td>
		    <td align=center>0.517</td>
		    <td align=center>0.133</td>
		    <td align=center>0.700</td>
            <td align=center><strong>0.102<strong/></td>
		    <td align=center>0.711</td>
		    <td align=center>0.191</td>
		    <td align=center>0.758</td>
        </tr>
    </tbody>
</table>

#### Music Probe codebook0
<table border="1" >
    
   <thead>
        <tr>
            <th rowspan="3">Codec</th>
            <th colspan="19">Dataset</th>
        </tr>
        <tr>
            <th colspan="2">emomusic</th>
            <th colspan="1">GTZAN</th>
            <th colspan="2">MTT</th>
            <th colspan="1">NSynthI</th>
		    <th colspan="1">VocalSetSinger</th>
		    <th colspan="1">VocalSetTech</th>
		    <th colspan="1">GS</th>
		    <th colspan="2">MTGInstrument</th>
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
			<td align=center>AP</td>
			<td align=center>AUCROC</td>
			<td align=center>AP</td>
			<td align=center>AUCROC</td>
    </thead>
    <tbody>
        <tr>
            <td align=center>DAC</td>
            <td align=center>0.354</td>
            <td align=center>0.000</td>
            <td align=center>0.600</td>
            <td align=center>0.175</td>
			<td align=center>0.741</td>
			<td align=center>0.563</td>
			<td align=center>0.226</td>
			<td align=center>0.315</td>
            <td align=center>0.088</td>
		    <td align=center>0.117</td>
		    <td align=center>0.638</td>
		    <td align=center>0.135</td>
		    <td align=center>0.690</td>
        </tr>
        <tr>
            <td align=center>Encodec</td>
            <td align=center><strong>0.465<strong/></td>
            <td align=center>0.092</td>
            <td align=center>0.543</td>
            <td align=center>0.119</td>
		    <td align=center>0.681</td>
		    <td align=center>0.563</td>
		    <td align=center>0.086</td>
		    <td align=center>0.268</td>
            <td align=center>0.088</td>
		    <td align=center>0.110</td>
		    <td align=center>0.630</td>
		    <td align=center>0.136</td>
		    <td align=center>0.701</td>
        </tr>
        <tr>
            <td align=center>SemantiCodec</td>
            <td align=center>0.456</td>
            <td align=center>0.267</td>
            <td align=center><strong>0.629<strong/></td>
            <td align=center>0.227</td>
		    <td align=center>0.825</td>
		    <td align=center><strong>0.625<strong/></td>
		    <td align=center>0.134</td>
		    <td align=center>0.477</td>
            <td align=center>0.229</td>
		    <td align=center><strong>0.150<strong/></td>
		    <td align=center><strong>0.724<strong/></td>
		    <td align=center><strong>0.224<strong/></td>
		    <td align=center><strong>0.793<strong/></td>
        </tr>
        <tr>
            <td align=center>Xcodec</td>
            <td align=center>0.375</td>
            <td align=center><strong>0.461<strong/></td>
            <td align=center>0.628</td>
            <td align=center><strong>0.261<strong/></td>
		    <td align=center><strong>0.838<strong/></td>
		    <td align=center>0.611</td>
		    <td align=center>0.320</td>
		    <td align=center><strong>0.488<strong/></td>
			<td align=center><strong>0.389<strong/></td>
		    <td align=center>0.140</td>
		    <td align=center>0.669</td>
		    <td align=center>0.191</td>
		    <td align=center>0.755</td>
        </tr>
	<tr>
            <td align=center>YuE</td>
            <td align=center>0.439</td>
            <td align=center>0.085</td>
            <td align=center>0.616</td>
            <td align=center>0.249</td>
		    <td align=center>0.831</td>
		    <td align=center>0.623</td>
		    <td align=center><strong>0.335<strong/></td>
		    <td align=center>0.475</td>
            <td align=center>0.346</td>
		    <td align=center>0.133</td>
		    <td align=center>0.670</td>
		    <td align=center>0.191</td>
		    <td align=center>0.758</td>
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
            <th colspan="2">Common_Voice</th>
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
            <td align=center>DAC</td>
			<td align=center>quantized_emb</td>
            <td align=center>0.526</td>
            <td align=center>0.229</td>
            <td align=center>0.535</td>
            <td align=center>0.483</td>
	   		<td align=center>0.325</td>
        </tr>
        <tr>
            <td align=center>Encodec</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.503</td>
            <td align=center>0.209</td>
		    <td align=center>0.574</td>
		    <td align=center>0.481</td>
		    <td align=center>0.275</td>
        </tr>
        <tr>
            <td align=center>SemantiCodec</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.490</td>
            <td align=center>0.200</td>
            <td align=center>0.723</td>
            <td align=center>0.482</td>
	    	<td align=center>0.620</td>
        </tr>
        <tr>
            <td align=center>WavTokenizer</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.582</td>
            <td align=center>0.288</td>
            <td align=center>0.524</td>
            <td align=center>0.484</td>
	    	<td align=center>0.135</td>
        </tr>
	<tr>
	    <td align=center>Mimi</td>
            <td align=center>quantized_emb</td>
            <td align=center><strong>0.442<strong/></td>
            <td align=center><strong>0.168<strong/></td>
            <td align=center>0.833</td>
            <td align=center>0.481</td>
	   		<td align=center>0.335</td>
        </tr>
	<tr>
	    <td align=center>SpeechTokenizer</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.469</td>
            <td align=center>0.190</td>
            <td align=center>0.776</td>
            <td align=center>0.498</td>
	    	<td align=center>0.670</td>
        </tr>
	<tr>
	    <td align=center>Xcodec</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.474</td>
            <td align=center>0.188</td>
            <td align=center>0.731</td>
            <td align=center>0.491</td>
            <td align=center>0.640</td>
        </tr>
	<tr>
	    <td align=center>YuE</td>
            <td align=center>quantized_emb</td>
            <td align=center>0.472</td>
            <td align=center>0.187</td>
            <td align=center>0.782</td>
            <td align=center>0.515</td>
            <td align=center>0.640</td>
        </tr>
	<tr>
	    <td align=center>hubert</td>
            <td align=center>unquantized_emb</td>
            <td align=center>-</td>
            <td align=center>-</td>
            <td align=center>0.877</td>
            <td align=center>0.495</td>
            <td align=center>0.525</td>
	</tr>
	<tr>
	    <td align=center>qwen2audioencoder</td>
            <td align=center>unquantized_emb</td>
            <td align=center>-</td>
            <td align=center>-</td>
            <td align=center><strong>0.953<strong/></td>
            <td align=center><strong>0.590<strong/></td>
            <td align=center><strong>0.975<strong/></td>
	</tr>
        </tr>
    </tbody>
</table>

#### Speech and Sound Probe codebook0
<table border="1" >
    
   <thead>
        <tr>
            <th rowspan="3">Codec</th>
            <th colspan="12">Dataset</th>
        </tr>
        <tr>
            <th colspan="1">Vocalsound</th>
            <th colspan="1">MELD</th>
            <th colspan="1">ESC50</th>
        </tr>
	<tr>	
			<td align=center>Acc</td>
            <td align=center>Acc</td>
            <td align=center>Acc</td>
    </thead>
    <tbody>
        <tr>
            <td align=center>DAC</td>
            <td align=center>0.511</td>
            <td align=center>0.481</td>
            <td align=center>0.285</td>
        </tr>
        <tr>
            <td align=center>Encodec</td>
            <td align=center>0.479</td>
            <td align=center>0.481</td>
            <td align=center>0.230</td>
        </tr>
        <tr>
            <td align=center>SemantiCodec</td>
            <td align=center>0.646</td>
            <td align=center>0.482</td>
            <td align=center>0.465</td>
        </tr>
	<tr>
	    <td align=center>Mimi</td>
            <td align=center><strong>0.794<strong/></td>
            <td align=center>0.481</td>
            <td align=center>0.265</td>
        </tr>
	<tr>
	    <td align=center>SpeechTokenizer</td>
            <td align=center>0.698</td>
            <td align=center><strong>0.489<strong/></td>
            <td align=center>0.420</td>
        </tr>
	<tr>
	    <td align=center>Xcodec</td>
            <td align=center>0.656</td>
            <td align=center>0.487</td>
            <td align=center><strong>0.525<strong/></td>
        </tr>
	<tr>
	    <td align=center>YuE</td>
            <td align=center>0.684</td>
            <td align=center>0.481</td>
            <td align=center>0.515</td>
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
| YuE            |  52.7  |  18.3    | 29.6    | 37.7   | 52.6    | 74.3    | 89.8    | 95.3    | 90.3    |
| SpeechTokenizer|  24.2  |  2.4     | __12.4__    | __24.8__    | __33.6__    | __40.9__    | __46.0__    | __50.3__    | __52.8__    |
| Mimi           |  269.6 |  32.9    | 189.3   | 334.7   | 383.3   | 424.2   | 431.7   | 456.9   | 459.7   |
| SemamiCodec    |  __14.8__  |  __1.2__     | 191.0   |    -    |    -    |    -    |    -    |    -    |    -    |

#### Emilia_EN(100ksteps)

| Codec          | ppl↓   | cb1_ppl  | cb2_ppl | cb3_ppl | cb4_ppl | cb5_ppl | cb6_ppl | cb7_ppl | cb8_ppl |
|----------------|--------|----------|---------|---------|---------|---------|---------|---------|---------|
| DAC            | 247    |  20.6    |  146.7  |  218    |  315.1  |  395.9  |  482.9  |  569.6  |  628.2  |
| Encodec        | 75.7   |  14.8    |  33.4   |  59     |  88.7   |  111.3  |  138.4  |  158.5  |  172.6  |
| WavTokenizer   | 104.7  |  104.7   |    -    |    -    |    -    |    -    |    -    |    -    |    -    |
| X-Codec        | 30.3   |  10.0    |  12.7   |  20.2   |  30.6   |  41.9   |  50.7   |  61.6   |  71.4   |
| YuE            | 29.0   |  9.3     |  16.0   |  19.9   |  29.3   |  38.7   |  51.0   |  55.2   |  54.1   |         
| SpeechTokenizer| 13.5   |  1.9     |  __5.5__    |  __12.1__   |  __18.3__   |  __22.3__   |  __25.1__   |  __28.6__   |  __30.8__   |
| Mimi           | 126.9  |  9.1     |  58.2   |  148.0  |  185.0  |  228.7  |  256.6  |  278.9  |  298.5  |        
| SemamiCodec    | __7.9__    |  __1.0__     |  82.1   |    -    |    -    |    -    |    -    |    -    |    -    |

#### MTG-Jamendo(100ksteps)

| Codec          | ppl↓  | cb1_ppl | cb2_ppl | cb3_ppl | cb4_ppl | cb5_ppl | cb6_ppl | cb7_ppl | cb8_ppl |
|----------------|-------|---------|---------|---------|---------|---------|---------|---------|---------|
| DAC            | 194   |  28.6   |  122.8  |  152.4  |  212.8  |  270.7  |  352.9  |  413.4  |  473.5  |
| Encodec        | 141.3 |  17.6   |  62.5   |  110.7  |  170    |  225.9  |  287    |  336.8  |  375.6  |
| WavTokenizer   | 38.2  |  38.2   |    -    |    -    |    -    |    -    |    -    |    -    |    -    |
| X-Codec        | 47.5  |  20.4   |  __19.6__   |  32.4   |  51.1   |  64.5   |  __74.5__   |  86.8   |  100.2  |
| YuE            | 46.2  |  18.3   |  28.7   |  __30.4__   |  __48.2__   |  __60.0__   |  74.9   |  __83.0__   |  __76.3__   |                
| SemamiCodec    | __15.5__  |  __1.0__    |  272.4  |    -    |    -    |    -    |    -    |    -    |    -    |

## Acknowledgement
We would like to extend a special thanks to authors of <a href="https://github.com/lucadellalib/audiocodecs">https://github.com/lucadellalib/audiocodecs</a> and <a href="https://github.com/a43992899/MARBLE">Marble</a>. Their work has been a great source of inspiration for us.

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
