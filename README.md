# Codec-Evaluation

## Purpose

1. how to evaluate the quality of codebook
2. collect all existing metrics

## Env Build

conda create -n codec_eval python==3.10 -y

conda activate codec_eval

git clone https://github.com/wuzhiyue111/Codec-Evaluation.git

cd Codec-Evaluation

bash env_build.sh

## Road Map

- [ ] multi codec deploy
    - multi codec deploy reference: https://github.com/lucadellalib/audiocodecs
- [ ] clean different dataset in marble benchmark
    - add code to redeploy marble in our benchmark
    - add code in marble base to evaluate our index 
- [ ] define the evaluation metrics of codec, codebooks
    - test ID sensitive in same semantic
    - 

## Probe Experiment

<table border="1" >
    
   <thead>
        <tr>
            <th rowspan="2">Codec</th>
            <th colspan="7">Dataset</th>
        </tr>
        <tr>
            <th>emomusic</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
        </tr>
    </thead>
	<tbody>
        <tr>
            <td align=center>DAC</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
        </tr>
        <tr>
            <td align=center>Encodec</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
        </tr>
        <tr>
            <td align=center>Semanticodec</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
        </tr>
        <tr>
            <td align=center>Wavtokenizer</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
        </tr>
        <tr>
            <td align=center>Mimi</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
        </tr>
        <tr>
            <td align=center>Speechtokenizer</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
        </tr>
        <tr>
            <td align=center>Wavlm_kmeans</td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
            <td align=center> </td>
        </tr>
    </tbody>
</table>


