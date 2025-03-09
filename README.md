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
            <td align=center>0.49</td>
            <td align=center>0.29</td>
            <td align=center>0.319</td>
            <td align=center>0.806</td>
	    <td align=center>0.226</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.45</td>
	    <td align=center>0.44</td>
            <td align=center>0.07</td>
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
            <td align=center>0.57</td>
            <td align=center>0.21</td>
            <td align=center>0.308</td>
            <td align=center>0.787</td>
            <td align=center>0.19</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.26</td>
	    <td align=center>0.40</td>
	    <td align=center>0.10</td>
	    <td align=center>0.60</td>
	    <td align=center>0.11</td>
        </tr>
        <tr>
            <td align=center rowspan="2">Semanticodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center>0.33</td>
            <td align=center>0.15</td>
            <td align=center>0.291</td>
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
            <td align=center>0.50</td>
            <td align=center>0.15</td>
            <td align=center>0.313</td>
            <td align=center></td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.25</td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
        <tr>
            <td align=center rowspan="2">Wavtokenizer</td>
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
            <td align=center>0.49</td>
            <td align=center>0.05</td>
            <td align=center>0.324</td>
            <td align=center>0.781</td>
            <td align=center>0.178</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.23</td>
	    <td align=center>0.56</td>
	    <td align=center>0.11</td>
            <td align=center> </td>
	    <td align=center> </td>
        </tr>
    </tbody>
</table>

### Speech Probe

## Probe Mutual Information

<table border="1" >
    
   <thead>
        <tr>
            <th rowspan="2">Codec</th>
	    <th rowspan="2">Mode</th>
            <th colspan="14">Dataset</th>
        </tr>
        <tr>
            <th colspan="2">Dac</th>
            <th colspan="2">Encodec</th>
            <th colspan="2">Mimi</th>
            <th colspan="2">Semanticodec</th>
            <th colspan="2">Speechtokenizer</th>
	    <th colspan="2">Wavlm_kmeans</th>
	    <th colspan="2">Wavtokenizer</th>
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
            <td align=center>0.49</td>
            <td align=center>0.29</td>
            <td align=center>0.319</td>
            <td align=center>0.806</td>
	    <td align=center>0.226</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.45</td>
	    <td align=center>0.44</td>
            <td align=center>0.07</td>
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
            <td align=center>0.57</td>
            <td align=center>0.21</td>
            <td align=center>0.308</td>
            <td align=center>0.787</td>
            <td align=center>0.19</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.26</td>
	    <td align=center>0.40</td>
	    <td align=center>0.10</td>
	    <td align=center>0.60</td>
	    <td align=center>0.11</td>
        </tr>
        <tr>
            <td align=center rowspan="2">Semanticodec</td>
            <td align=center>unquantized_emb</td>
            <td align=center>0.33</td>
            <td align=center>0.15</td>
            <td align=center>0.291</td>
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
            <td align=center>0.50</td>
            <td align=center>0.15</td>
            <td align=center>0.313</td>
            <td align=center></td>
            <td align=center> </td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.25</td>
	    <td align=center> </td>
	    <td align=center></td>
	    <td align=center> </td>
	    <td align=center> </td>
        </tr>
        <tr>
            <td align=center rowspan="2">Wavtokenizer</td>
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
            <td align=center>0.49</td>
            <td align=center>0.05</td>
            <td align=center>0.324</td>
            <td align=center>0.781</td>
            <td align=center>0.178</td>
	    <td align=center> </td>
	    <td align=center> </td>
	    <td align=center>0.23</td>
	    <td align=center>0.56</td>
	    <td align=center>0.11</td>
            <td align=center> </td>
	    <td align=center> </td>
        </tr>
    </tbody>
</table>


