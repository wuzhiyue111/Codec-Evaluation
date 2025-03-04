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

## Probe Experiment
### Marble Probe (Music)
<table border="1" >
    
   <thead>
        <tr>
            <th rowspan="3">Codec</th>
	    <th rowspan="3">Mode</th>
            <th colspan="7">Dataset</th>
        </tr>
        <tr>
            <th colspan="2">emomusic</th>
            <th colspan="2">GTZAN</th>
            <th></th>
            <th></th>
            <th></th>
	    <th></th>
        </tr>
	<tr>
	    <td align=center>A</td>
            <td align=center>V</td>
	    <td align=center>Acc</td>
            <td align=center>Prec</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
    </thead>
    <tbody>
        <tr>
            <td align=center rowspan="2">DAC</td>
            <td align=center>unquantized_emb</td>
            <td align=center>0.51</td>
            <td align=center>0.34</td>
            <td align=center> </td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.49</td>
            <td align=center>0.29</td>
            <td align=center>0.319</td>
            <td align=center>0.319</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
        <tr>
            <td align=center rowspan="2">Encodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center>0.56</td>
            <td align=center>0.24</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.57</td>
            <td align=center>0.21</td>
            <td align=center>0.253</td>
            <td align=center>0.253</td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
        <tr>
            <td align=center rowspan="2">Semanticodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center>0.33</td>
            <td align=center>0.15</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.50</td>
            <td align=center>0.15</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
        <tr>
            <td align=center rowspan="2">Wavtokenizer</td>
            <td align=center>unquantized_emb</td>
            <td align=center>0.49</td>
            <td align=center>0.05</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
	<tr>
            <td align=center>quantized_emb</td>
            <td align=center>0.49</td>
            <td align=center>0.05</td>
            <td align=center>0.297</td>
            <td align=center>0.297</td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
    </tbody>
</table>

### Speech Probe


