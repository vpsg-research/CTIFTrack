# CTIFTrack
Once the paper is accepted for publication, we will open the source code to the public immediately.

## Performance
![](imgs/aucVSfps.svg)

Highlight 1:TIF module:

lib.models.ctiftrack.TIF

Highlight 2:OTFR module:

lib.models.ctiftrack.vit_cae_async

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-c3ow" colspan="3">Accuracy</th>
    <th class="tg-c3ow" colspan="5">FPS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-c3ow">GOT-10k</td>
    <td class="tg-c3ow">LaSOT</td>
    <td class="tg-c3ow">TrackingNet</td>
    <td class="tg-c3ow"> Nvidia Tesla T4-16GB</td>
    
  </tr>
  <tr>
    <td class="tg-0pky">ctiftrack</td>
    <td class="tg-c3ow">71.3</td>
    <td class="tg-c3ow">66.9</td>
    <td class="tg-c3ow">82.3</td>
    <td class="tg-c3ow">71.98</td>

  </tr>
  
</tbody>
</table>
