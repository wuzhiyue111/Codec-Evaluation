# Codec-Evaluation

## Purpose

1. how to evaluate the quality of codebook
2. collect all existing metrics for reconstruction
3. collect all existing metrics for Linear Probing (Music and Speech)

## Env Build

conda create -n codec_eval python==3.10 -y

conda activate codec_eval

git clone https://github.com/wuzhiyue111/Codec-Evaluation.git

cd Codec-Evaluation

bash env_build.sh

## Road Map

- [x] multi codec deploy
    - multi codec deploy reference: https://github.com/lucadellalib/audiocodecs
- [ ] clean different dataset in marble benchmark
    - add code to redeploy marble in our benchmark
    - add code in marble base to evaluate our index 
- [ ] define the evaluation metrics of codec, codebooks
    - test ID sensitive in same semantic
    - 

## Reconstruction Metric
### Speech
<table border="1" align="center">
  <thead>
    <tr>
      <th rowspan="2">Codec</th>
      <th colspan="5">Metrics</th>
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
  </tbody>
</table>

### Music
<table border="1" align="center">
  <thead>
    <tr>
      <th rowspan="2">Codec</th>
      <th colspan="5">Metrics</th>
    </tr>
    <tr>
      <th>PESQ</th>
      <th>Speaker_Sim</th>
      <th>WER</th>
      <th>CER</th>
      <th>STOI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DAC</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Encodec</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Mimi</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>SemantiCodec</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>WavTokenizer</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>SpeechTokenizer</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
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
            <th colspan="12">Dataset</th>
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
	    <th colspan="2">MTGInstrument</th>
        </tr>
	<tr>
	    <td align=center>A</td>
            <td align=center>V</td>
	    <td align=center>Acc</td>
            <td align=center>AUROC</td>
	    <td align=center>Ap</td>
	    <td align=center>Acc</td>
	    <td align=center>Acc</td>
	    <td align=center>Acc</td>
	    <td align=center>Acc</td>
	    <td align=center>Acc</td>
	    <td align=center>ROC</td>
	    <td align=center>AP</td>
    </thead>
    <tbody>
        <tr>
            <td align=center rowspan="2">DAC</td>
            <td align=center>unquantized_emb</td>
            <td align=center>0.51</td>
            <td align=center>0.34</td>
            <td align=center>0.321</td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.40</td>
	    <td align=center>0.41</td>
            <td align=center>0.09</td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.47</td>
            <td align=center>0.06</td>
            <td align=center>0.345</td>
            <td align=center>0.806</td>
	    <td align=center>0.226</td>
	    <td align=center>0.604</td>
	    <td align=center> </td>
	    <td align=center>0.419</td>
	    <td align=center>0.349</td>
            <td align=center>0.088</td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
        <tr>
            <td align=center rowspan="2">Encodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center>0.56</td>
            <td align=center>0.24</td>
            <td align=center>0.314</td>
            <td align=center>0.792</td>
            <td align=center>0.196</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.35</td>
	    <td align=center>0.41</td>
	    <td align=center>0.10</td>
	    <td align=center>0.60</td>
	    <td align=center>0.11</td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.48</td>
            <td align=center>0.08</td>
            <td align=center>0.341</td>
            <td align=center>0.787</td>
            <td align=center>0.19</td>
	    <td align=center>0.543</td>
	    <td align=center> </td>
	    <td align=center>0.299</td>
	    <td align=center>0.301</td>
	    <td align=center>0.088</td>
	    <td align=center>0.60</td>
	    <td align=center>0.11</td>
        </tr>
        <tr>
            <td align=center rowspan="2">SemantiCodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center>0.55</td>
            <td align=center>0.35</td>
            <td align=center>0.29</td>
            <td align=center></td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>\
	    <td align=center>0.16</td>
	    <td align=center> </td>
	    <td align=center>0.06</td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.48</td>
            <td align=center>0.31</td>
            <td align=center>0.501</td>
            <td align=center></td>
            <td align=center> </td>
	    <td align=center>0.658</td>
	    <td align=center> </td>
	    <td align=center>0.344</td>
	    <td align=center>0.451</td>
	    <td align=center>0.328</td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
        <tr>
            <td align=center rowspan="2">WavTokenizer</td>
            <td align=center>unquantized_emb</td>
            <td align=center>0.49</td>
            <td align=center>0.05</td>
            <td align=center>0.322</td>
            <td align=center>0.782</td>
            <td align=center>0.179</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.24</td>
	    <td align=center>0.57</td>
	    <td align=center>0.11</td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.47</td>
            <td align=center>0.07</td>
            <td align=center>0.37</td>
            <td align=center>0.781</td>
            <td align=center>0.178</td>
	    <td align=center>0.541</td>
	    <td align=center> </td>
	    <td align=center>0.130</td>
	    <td align=center>0.287</td>
	    <td align=center>0.090</td>
            <td align=center> </td>
	    <td align=center> </td>
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
            <th colspan="1"></th>
            <th colspan="1"></th>
	    <th colspan="1"></th>
	    <th colspan="1"></th>
	    <th colspan="1"></th>
	    <th colspan="2"></th>
        </tr>
	<tr>
	    <td align=center>WER</td>
            <td align=center>CER</td>
	    <td align=center>Acc</td>
            <td align=center>Acc</td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
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
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.573</td>
            <td align=center>0.369</td>
            <td align=center>0.535</td>
            <td align=center>0.483</td>
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
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
	    <td align=center></td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center></td>
            <td align=center></td>
	    <td align=center>0.57</td>
	    <td align=center>0.481</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
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
	    <td align=center> </td>\
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center></td>
            <td align=center></td>
            <td align=center>0.723</td>
            <td align=center>0.485</td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
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
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.645</td>
            <td align=center>0.416</td>
            <td align=center>0.524</td>
            <td align=center>0.484</td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
            <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
	    <td align=center rowspan="2">Mimi</td>
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
	</tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.499</td>
            <td align=center>0.22</td>
            <td align=center>0.824</td>
            <td align=center>0.481</td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
            <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
	    <td align=center rowspan="2">SpeechTokenizer</td>
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
	</tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.433</td>
            <td align=center>0.164</td>
            <td align=center>0.772</td>
            <td align=center>0.498</td>
            <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center></td>
	    <td align=center></td>
            <td align=center> </td>
	    <td align=center> </td>
        </tr>
    </tbody>
</table>

## Mutual Information Estimation
<table border="1" align="center">
  <thead>
    <tr>
      <th rowspan="2">Task</th>
      <th colspan="6">Codec</th>
    </tr>
    <tr>
      <th>DAC</th>
      <th>Encodec</th>
      <th>Mimi</th>
      <th>SemantiCodec</th>
      <th>SpeechTokenizer</th>
      <th>WavTokenizer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>相同位置id数(codebook)</td>
      <td>[247,21,2,1,0,0,0,0]</td>
      <td>[534,426,292,272,237,194,167,176]</td>
      <td>[85,61,47,44,33,31,43,33]</td>
      <td>[382,118]</td>
      <td>[412,334,273,236,198,175,131,128]</td>
      <td>[125]</td>    
    </tr>
    <tr>
      <td>最大公共子串长度(codebook)</td>
      <td>[9,3,1,1,0,0,0,0]</td>
      <td>[18,12,8,6,5,5,5,4]</td>
      <td>[11,10,6,3,3,3,5,9]</td>
      <td>[21,5]</td>
      <td>[74,15,8,21,7,6,5,4]</td>
      <td>[7]</td>
    </tr>
    <tr>
      <td>偏移2ms相同位置id数(codebook)</td>
      <td>[149,10,5,4,3,2,1]</td>
      <td>[334,221,133,114,93,72,72,67]</td>
      <td>[105,82,48,39,31,15,33,17]</td>
      <td>[438,214]</td>
      <td>[407,214,164,123,129,92,60,77]</td>
      <td>[124]</td>
    </tr>
    <tr>
      <td>偏移2ms最大公共子串长度(codebook)</td>
      <td>[6,1,1,1,1,1,1,1]</td>
      <td>[16,7,5,4,3,3,4,3]</td>
      <td>[12,26,11,8,7,4,6,1]</td>
      <td>[126,8]</td>
      <td>[33,12,8,18,8,4,4,6]</td>
      <td>[10]</td>
    </tr>
    <tr>
      <td>§§</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <th></th>
    </tr>
    <tr>
      <td>§§</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>






