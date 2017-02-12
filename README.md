# ngsapps
This repository contains applications for the [NGSolve finite element library](https://gitlab.asc.tuwien.ac.at/jschoeberl/ngsolve-docu/wikis/home).

## Models
<table>
<tr>
<td>The <a href="https://github.com/jfpxtal/ngsapps/wiki/surfacebulk">surfacebulk wiki page</a> demonstrates the Trace() operator, symbolic definition of bilinear forms and Euler time stepping.</td>
<td><img src="https://raw.githubusercontent.com/jfpxtal/ngsapps/wiki/surfacebulk.gif" width=200px/></td>
</tr>
<tr>
<td>
<a href="https://github.com/jfpxtal/ngsapps/blob/master/cahnhilliard.py">cahnhilliard.py</a> solves the <a href="https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation">Cahn-Hilliard equation</a> for phase separation.
<br><br>
<code>Newton solver</code>
</td>
<td>...</td>
</tr>
<tr>
<td>The <a href="https://github.com/jfpxtal/ngsapps/tree/master/precip">precipitation</a> folder contains FDM and FEM implementations of a two-species reaction-diffusion system (<a href="http://www-users.math.umn.edu/~scheel/preprints/pf0.pdf">paper</a>), which exhibits wave patterns as seen on the right. See <a href="https://www.youtube.com/watch?v=-AlpHiZdJdU&list=PLePJW8tlg_8buzdhBNOOSYeuxnNdZ8ZSl">here</a> for some videos.
<br><br>
<code>NumPy</code> <code>Matplotlib</code> <code>Newton solver</code> <code>DG</code>
</td>
<td><img src="https://raw.githubusercontent.com/alexschlueter/myngsolve/master/precip_ngsolve.gif" width=900px/></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/Bleb_(cell_biology)">Cellular blebs</a> are protrusions of the plasma membrane of a biological cell, which form during apoptosis (programmed cell death).  We <a href="https://github.com/jfpxtal/ngsapps/tree/master/bleb">implemented a model</a> to investigate the critical pressure under which a bleb forms. The model takes into account the nonlinear kinetics of binding proteins near the membrane.
<br><br>
<code>Matplotlib</code> <code>Newton solver</code>
</td>
<td>...</td>
</tr>
<tr>
<td><a href="https://github.com/jfpxtal/ngsapps/blob/master/crowdedtransport.py">crowdedtransport.py</a>
<br><br>
<code>Matplotlib</code> <code>Convolution</code> <code>DG</code>
</td>
<td>...</td>
</tr>
<tr>
<td><a href="https://github.com/jfpxtal/ngsapps/tree/master/crossdiffusion">crossdiffusion</a>
<br><br>
<code>Matplotlib + Qt</code> <code>Convolution</code> <code>ComposeCF</code> <code>DG</code>
</td>
<td>...</td>
</tr>
<tr>
<td><a href="https://github.com/jfpxtal/ngsapps/tree/master/nonlocal">nonlocaldiffusion</a>
<br><br>
<code>Convolution</code> <code>Periodic</code> <code>DG</code>
</td>
<td>...</td>
</tr>
</table>

## Utils
- LagrangeFESpace
- ParameterLF
- ComposeCF
- RandomCF
- CacheCF
- ZLogZCF

## Build instructions
