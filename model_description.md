1. Physical model
Let the dB-domain radio map be D∈RH×W×K\mathcal{D} \in \mathbb{R}^{H \times W \times K}
D∈RH×W×K over KK
K frequency bands {f1,…,fK}\{f_1, \ldots, f_K\}
{f1​,…,fK​}, with RR
R known emitter locations {tr}r=1R\{t_r\}_{r=1}^R
{tr​}r=1R​ extracted from the TXPOS input. For each emitter, define a deterministic distance map dr(x)=∥x−tr∥2d_r(x) = \|x - t_r\|_2
dr​(x)=∥x−tr​∥2​ and a building map B∈[0,1]H×WB \in [0,1]^{H \times W}
B∈[0,1]H×W.
The physics-grounded decomposition replacing the original "low-rank background + sparse foreground" is the SLF-PSD block-term tensor decomposition (Fu, Sidiropoulos et al.; Zhang–Fu–Wang–Zhao–Hong, IEEE TSP 2020):
D(x,k)  =  ∑r=1RSr(x) cr(k)⏟X :  physical structure  +  E(x,k)⏟sparse multipath residual  +  N(x,k)⏟observation noise\mathcal{D}(x, k) \;=\; \underbrace{\sum_{r=1}^R \mathbf{S}_r(x)\, \mathbf{c}_r(k)}_{\mathcal{X}\,:\;\text{physical structure}} \;+\; \underbrace{\mathcal{E}(x, k)}_{\text{sparse multipath residual}} \;+\; \underbrace{\mathcal{N}(x, k)}_{\text{observation noise}}D(x,k)=X:physical structurer=1∑R​Sr​(x)cr​(k)​​+sparse multipath residualE(x,k)​​+observation noiseN(x,k)​​
where Sr∈RH×W\mathbf{S}_r \in \mathbb{R}^{H \times W}
Sr​∈RH×W is the spatial loss field of emitter rr
r (its propagation pattern over space, capturing path loss + shadowing) and cr∈RK\mathbf{c}_r \in \mathbb{R}^{K}
cr​∈RK is its PSD across the bands. The mode-3 unfolding of X\mathcal{X}
X has rank exactly R≪KR \ll K
R≪K — the inter-frequency low-rankness now has a physical referent (one rank per emitter) rather than a generic nuclear-norm penalty.
2. Friis pre-conditioning
Free-space propagation contributes a deterministic frequency shift 20log⁡10(fk/f0)20\log_{10}(f_k/f_0)
20log10​(fk​/f0​) to every pixel. Folding this into cr\mathbf{c}_r
cr​ inflates the apparent rank of the frequency mode whenever multiple emitters have different reference frequencies. Strip it out once, up front:
D~(x,k)  =  D(x,k)  −  Fk,Fk  =  20log⁡10(fk/f0).\tilde{\mathcal{D}}(x, k) \;=\; \mathcal{D}(x, k) \;-\; F_k, \qquad F_k \;=\; 20\log_{10}(f_k / f_0).D~(x,k)=D(x,k)−Fk​,Fk​=20log10​(fk​/f0​).
After pre-conditioning, every Sr\mathbf{S}_r
Sr​ becomes frequency-independent and each cr\mathbf{c}_r
cr​ contains only the *non-Friis* part of the emitter spectrum (antenna pattern, hardware response, intentional shaping). For a flat-spectrum omnidirectional source, cr\mathbf{c}_r
cr​ is constant in kk
k — the cleanest possible low-rank structure in the frequency mode. All optimization is done on D~\tilde{\mathcal{D}}
D~; you add FkF_k
Fk​ back at the end.
3. Optimization problem
Combining BTD structure, sparse residual, noise relaxation, and a learned SLF prior, the radio map estimation problem becomes:
min⁡X, {Sr,cr}, E, N∑r=1Rgr(Sr)  +  λ∥E∥1\min_{\mathcal{X},\, \{\mathbf{S}_r, \mathbf{c}_r\},\, \mathcal{E},\, \mathcal{N}} \quad
\sum_{r=1}^R g_r(\mathbf{S}_r) \;+\; \lambda \|\mathcal{E}\|_1X,{Sr​,cr​},E,Nmin​r=1∑R​gr​(Sr​)+λ∥E∥1​
subject to
X  =  ∑r=1RSr∘cr,PΩ(X+E+N)=PΩ(D~),∥PΩ(N)∥F≤δ,\mathcal{X} \;=\; \sum_{r=1}^R \mathbf{S}_r \circ \mathbf{c}_r, \qquad
\mathcal{P}_\Omega(\mathcal{X} + \mathcal{E} + \mathcal{N}) = \mathcal{P}_\Omega(\tilde{\mathcal{D}}), \qquad
\|\mathcal{P}_\Omega(\mathcal{N})\|_F \leq \delta,X=r=1∑R​Sr​∘cr​,PΩ​(X+E+N)=PΩ​(D~),∥PΩ​(N)∥F​≤δ,
with Sr,cr≥0\mathbf{S}_r, \mathbf{c}_r \geq 0
Sr​,cr​≥0. Here grg_r
gr​ is a learned regulariser on the *r*-th SLF — this is the slot where the physics-aware proximal operator lives.
Note what's gone: there is no longer a sum-of-nuclear-norms term on the three mode unfoldings. The BTD constraint X=∑rSr∘cr\mathcal{X} = \sum_r \mathbf{S}_r \circ \mathbf{c}_r
X=∑r​Sr​∘cr​ enforces low-rankness in the frequency mode *exactly* (rank ≤ R), and spatial regularity is handled by the SLF prior grg_r
gr​.
4. The ray-cast / wave-propagation-inspired proximal operator
The learned proximal of grg_r
gr​ replaces your CBAM/FiLM block. Instead of feeding the SLF estimate through a generic CNN with environment features stuck on the side, the proximal operator's inputs are *physically meaningful per-emitter fields*. For emitter rr
r, define:
The distance map dr(x)=∥x−tr∥2d_r(x) = \|x - t_r\|_2
dr​(x)=∥x−tr​∥2​ — log-transformed and normalised before use.
The ray-cast shadowing field (the RadioUNet-style line integral):
Tr(x)  =  ∫01B((1−s) tr+s x) ds.\mathcal{T}_r(x) \;=\; \int_0^1 B\big((1-s)\,t_r + s\,x\big)\, ds.Tr​(x)=∫01​B((1−s)tr​+sx)ds.
For each pixel xx
x, this is the obstacle-density integrated along the straight TX→pixel ray. It's the geometric-optics shadowing term and is differentiable in BB
B. Discretise as a uniform sum over LL
L samples along each ray; in PyTorch this is one grid_sample call per emitter and is fast.
The deterministic free-space anchor:
Prfree(x)  =  − 10 n0log⁡10 ⁣(dr(x)/d0)  −  α0 Tr(x),\mathbf{P}_r^{\text{free}}(x) \;=\; -\,10\, n_0 \log_{10}\!\big(d_r(x)/d_0\big) \;-\; \alpha_0\, \mathcal{T}_r(x),Prfree​(x)=−10n0​log10​(dr​(x)/d0​)−α0​Tr​(x),
with n0,α0n_0, \alpha_0
n0​,α0​ learnable scalars (typical inits: n0=2.5n_0 = 2.5
n0​=2.5, α0=1\alpha_0 = 1
α0​=1). This is the closed-form prediction of log-distance-plus-line-integral shadowing — already physically reasonable, and what the network only has to *correct*.
The learned proximal operator at unrolling iteration kk
k for emitter rr
r is then:
Prk+1  =  Prfree  +  Vk( S~r,  dr,  Tr,  B,  Prfree )\mathbf{P}_r^{k+1} \;=\; \mathbf{P}_r^{\text{free}} \;+\; V_k\Big(\, \tilde{\mathbf{S}}_r,\; d_r,\; \mathcal{T}_r,\; B,\; \mathbf{P}_r^{\text{free}}\,\Big)Prk+1​=Prfree​+Vk​(S~r​,dr​,Tr​,B,Prfree​)
where S~r=Srk+1+Γrk/θ\tilde{\mathbf{S}}_r = \mathbf{S}_r^{k+1} + \Gamma_r^k / \theta
S~r​=Srk+1​+Γrk​/θ is the current SLF estimate plus its scaled dual, and VkV_k
Vk​ is a small CNN that outputs a *residual correction* to the free-space anchor. Parameters of VkV_k
Vk​ are shared across emitters (so the operator generalises to scenes with different RR
R). A zero-initialised final conv layer makes VkV_k
Vk​ start as the identity, so the network begins from the physics-only prediction and learns only the correction — same training-stability trick you used with FiLM.
5. ADMM derivation
Introduce auxiliary variables Pr=Sr\mathbf{P}_r = \mathbf{S}_r
Pr​=Sr​ (to apply the learned operator) and dual variables Λ\Lambda
Λ (data), Γr\Gamma_r
Γr​ (SLF-proxy), Y\mathcal{Y}
Y (BTD consistency). The augmented Lagrangian is:
L  =  ∑rgr(Pr)  +  λ∥E∥1+⟨Λ, PΩ(D~)−X−E−N⟩+μ2∥PΩ(D~)−X−E−N∥F2+∑r[⟨Γr, Sr−Pr⟩+θ2∥Sr−Pr∥F2]+⟨Y, X−∑rSr∘cr⟩+ρ2∥X−∑rSr∘cr∥F2.\begin{aligned}
\mathcal{L} \;=\;& \sum_r g_r(\mathbf{P}_r) \;+\; \lambda \|\mathcal{E}\|_1 \\
&+ \langle \Lambda,\, \mathcal{P}_\Omega(\tilde{\mathcal{D}}) - \mathcal{X} - \mathcal{E} - \mathcal{N}\rangle + \tfrac{\mu}{2}\|\mathcal{P}_\Omega(\tilde{\mathcal{D}}) - \mathcal{X} - \mathcal{E} - \mathcal{N}\|_F^2 \\
&+ \sum_r \Big[\langle \Gamma_r,\, \mathbf{S}_r - \mathbf{P}_r\rangle + \tfrac{\theta}{2}\|\mathbf{S}_r - \mathbf{P}_r\|_F^2\Big] \\
&+ \langle \mathcal{Y},\, \mathcal{X} - \textstyle\sum_r \mathbf{S}_r \circ \mathbf{c}_r\rangle + \tfrac{\rho}{2}\big\|\mathcal{X} - \textstyle\sum_r \mathbf{S}_r \circ \mathbf{c}_r\big\|_F^2.
\end{aligned}L=​r∑​gr​(Pr​)+λ∥E∥1​+⟨Λ,PΩ​(D~)−X−E−N⟩+2μ​∥PΩ​(D~)−X−E−N∥F2​+r∑​[⟨Γr​,Sr​−Pr​⟩+2θ​∥Sr​−Pr​∥F2​]+⟨Y,X−∑r​Sr​∘cr​⟩+2ρ​​X−∑r​Sr​∘cr​​F2​.​
Each ADMM iteration cycles through closed-form (or learned) updates of the seven variable groups.
X\mathcal{X}
X update — quadratic, closed form. Setting ∂L/∂X=0\partial\mathcal{L}/\partial\mathcal{X} = 0
∂L/∂X=0:
Xk+1  =  μ ΨX  +  ρ ΨBTDμ+ρ,\mathcal{X}^{k+1} \;=\; \frac{\mu\,\Psi_{\mathcal{X}} \;+\; \rho\,\Psi_{\text{BTD}}}{\mu + \rho},Xk+1=μ+ρμΨX​+ρΨBTD​​,
with
ΨX=PΩ(D~)−Ek−Nk+1μΛk,ΨBTD=∑rSrk∘crk−1ρYk.\Psi_{\mathcal{X}} = \mathcal{P}_\Omega(\tilde{\mathcal{D}}) - \mathcal{E}^k - \mathcal{N}^k + \tfrac{1}{\mu}\Lambda^k,
\qquad
\Psi_{\text{BTD}} = \sum_r \mathbf{S}_r^k \circ \mathbf{c}_r^k - \tfrac{1}{\rho}\mathcal{Y}^k.ΨX​=PΩ​(D~)−Ek−Nk+μ1​Λk,ΨBTD​=r∑​Srk​∘crk​−ρ1​Yk.
This replaces both your old X\mathcal{X}
X-update *and* the three Mi\mathcal{M}_i
Mi​-SVT blocks. No SVD needed anywhere — the BTD constraint absorbs the low-rank role.
Sr\mathbf{S}_r
Sr​ update — closed form per emitter. Holding all Sr′≠r\mathbf{S}_{r' \ne r}
Sr′=r​ fixed, define the residual after subtracting the contribution of other emitters:
Rrk  =  Xk+1+1ρYk  −  ∑r′≠rSr′k∘cr′k.\mathcal{R}_r^k \;=\; \mathcal{X}^{k+1} + \tfrac{1}{\rho}\mathcal{Y}^k \;-\; \sum_{r' \ne r} \mathbf{S}_{r'}^k \circ \mathbf{c}_{r'}^k.Rrk​=Xk+1+ρ1​Yk−r′=r∑​Sr′k​∘cr′k​.
Setting ∂L/∂Sr=0\partial\mathcal{L}/\partial\mathbf{S}_r = 0
∂L/∂Sr​=0 gives a H×WH \times W
H×W linear system that decouples per pixel (because cr\mathbf{c}_r
cr​ multiplies Sr\mathbf{S}_r
Sr​ rank-1 in the frequency mode):
Srk+1  =  ρ ⟨Rrk, crk⟩3  +  θ(Prk−1θΓrk)ρ ∥crk∥22  +  θ,\mathbf{S}_r^{k+1} \;=\; \frac{\rho\, \langle \mathcal{R}_r^k,\, \mathbf{c}_r^k\rangle_3 \;+\; \theta\big(\mathbf{P}_r^k - \tfrac{1}{\theta}\Gamma_r^k\big)}{\rho\,\|\mathbf{c}_r^k\|_2^2 \;+\; \theta},Srk+1​=ρ∥crk​∥22​+θρ⟨Rrk​,crk​⟩3​+θ(Prk​−θ1​Γrk​)​,
where ⟨R,c⟩3=∑k=1KR(:,:,k) c(k)\langle\mathcal{R}, \mathbf{c}\rangle_3 = \sum_{k=1}^K \mathcal{R}(:,:,k)\, c(k)
⟨R,c⟩3​=∑k=1K​R(:,:,k)c(k) is the mode-3 contraction. The cost is one inner product across frequency per pixel — cheaper than your current SVT.
cr\mathbf{c}_r
cr​ update — closed form per emitter, per band. Symmetric to the above:
crk+1(k)  =  max⁡ ⁣(0, ⟨Rrk(:,:,k), Srk+1⟩F∥Srk+1∥F2),\mathbf{c}_r^{k+1}(k) \;=\; \max\!\left(0,\, \frac{\big\langle \mathcal{R}_r^{k}(:,:,k),\, \mathbf{S}_r^{k+1}\big\rangle_F}{\|\mathbf{S}_r^{k+1}\|_F^2}\right),crk+1​(k)=max(0,∥Srk+1​∥F2​⟨Rrk​(:,:,k),Srk+1​⟩F​​),
with the max⁡\max
max enforcing non-negativity. This is a single inner product over space per emitter per band — essentially free.
Pr\mathbf{P}_r
Pr​ update — learned proximal (this is where the network does its work). As given in §4:
Prk+1  =  Prfree+Vk ⁣(Srk+1+1θΓrk,  dr,  Tr,  B,  Prfree).\mathbf{P}_r^{k+1} \;=\; \mathbf{P}_r^{\text{free}} + V_k\!\Big(\mathbf{S}_r^{k+1} + \tfrac{1}{\theta}\Gamma_r^k,\; d_r,\; \mathcal{T}_r,\; B,\; \mathbf{P}_r^{\text{free}}\Big).Prk+1​=Prfree​+Vk​(Srk+1​+θ1​Γrk​,dr​,Tr​,B,Prfree​).
The proximal CNN VkV_k
Vk​ is the *only* learned tensor-shaped module in the iteration. It is per-emitter but shares weights across emitters in a single iteration; weights differ across iterations k=1,…,Kiterk = 1, \ldots, K_\text{iter}
k=1,…,Kiter​. The free-space anchor is computed once per sample and reused at every iteration — no extra cost.
E\mathcal{E}
E update — soft threshold, unchanged in form.
Ek+1  =  sign(ΨE) ⊙ max⁡ ⁣(∣ΨE∣−λμ, 0),ΨE=PΩ(D~)−Xk+1−Nk+1μΛk.\mathcal{E}^{k+1} \;=\; \text{sign}(\Psi_{\mathcal{E}})\,\odot\, \max\!\big(|\Psi_{\mathcal{E}}| - \tfrac{\lambda}{\mu},\, 0\big),\qquad
\Psi_{\mathcal{E}} = \mathcal{P}_\Omega(\tilde{\mathcal{D}}) - \mathcal{X}^{k+1} - \mathcal{N}^k + \tfrac{1}{\mu}\Lambda^k.Ek+1=sign(ΨE​)⊙max(∣ΨE​∣−μλ​,0),ΨE​=PΩ​(D~)−Xk+1−Nk+μ1​Λk.
N\mathcal{N}
N update — noise projection, unchanged from your derivation.
Nk+1  =  PΩC(ΨN)  +  min⁡ ⁣{δ∥PΩ(ΨN)∥F, 1} PΩ(ΨN),\mathcal{N}^{k+1} \;=\; \mathcal{P}_{\Omega^C}(\Psi_{\mathcal{N}}) \;+\; \min\!\Big\{\tfrac{\delta}{\|\mathcal{P}_\Omega(\Psi_{\mathcal{N}})\|_F},\, 1\Big\}\,\mathcal{P}_\Omega(\Psi_{\mathcal{N}}),Nk+1=PΩC​(ΨN​)+min{∥PΩ​(ΨN​)∥F​δ​,1}PΩ​(ΨN​),
with ΨN=PΩ(D~)−Xk+1−Ek+1+1μΛk\Psi_{\mathcal{N}} = \mathcal{P}_\Omega(\tilde{\mathcal{D}}) - \mathcal{X}^{k+1} - \mathcal{E}^{k+1} + \tfrac{1}{\mu}\Lambda^k
ΨN​=PΩ​(D~)−Xk+1−Ek+1+μ1​Λk.
Multiplier updates.
Λk+1=Λk+μ(PΩ(D~)−Xk+1−Ek+1−Nk+1),\Lambda^{k+1} = \Lambda^k + \mu\big(\mathcal{P}_\Omega(\tilde{\mathcal{D}}) - \mathcal{X}^{k+1} - \mathcal{E}^{k+1} - \mathcal{N}^{k+1}\big),Λk+1=Λk+μ(PΩ​(D~)−Xk+1−Ek+1−Nk+1),
Γrk+1=Γrk+θ(Srk+1−Prk+1),Yk+1=Yk+ρ ⁣(Xk+1−∑rSrk+1∘crk+1).\Gamma_r^{k+1} = \Gamma_r^k + \theta\big(\mathbf{S}_r^{k+1} - \mathbf{P}_r^{k+1}\big),\qquad
\mathcal{Y}^{k+1} = \mathcal{Y}^k + \rho\!\left(\mathcal{X}^{k+1} - \sum_r \mathbf{S}_r^{k+1} \circ \mathbf{c}_r^{k+1}\right).Γrk+1​=Γrk​+θ(Srk+1​−Prk+1​),Yk+1=Yk+ρ(Xk+1−r∑​Srk+1​∘crk+1​).
Final reconstruction. After KiterK_\text{iter}
Kiter​ unrolled iterations, undo the Friis pre-conditioning:
D^(x,k)  =  XKiter(x,k)  +  EKiter(x,k)  +  Fk.\hat{\mathcal{D}}(x, k) \;=\; \mathcal{X}^{K_\text{iter}}(x, k) \;+\; \mathcal{E}^{K_\text{iter}}(x, k) \;+\; F_k.D^(x,k)=XKiter​(x,k)+EKiter​(x,k)+Fk​.
6. The unrolled network at a glance
One unrolling block corresponds to one ADMM sweep through the seven updates above. Per block, the learnable parameters are:
The scalars μ,ρ,θ,β,λ,δ\mu, \rho, \theta, \beta, \lambda, \delta
μ,ρ,θ,β,λ,δ (kept positive via softplus, as in your current implementation), plus the two physics scalars n0,α0n_0, \alpha_0
n0​,α0​ in the free-space anchor — these get learned values that drift away from their physical initialisations only as far as the data demands. Watching them after training is itself a sanity check: if n0n_0
n0​ trains to 2.7 for urban data, that's exactly what a comms textbook would predict.
The CNN weights of VkV_k
Vk​ — a small residual CNN (a few conv layers with the same spatial dimensions throughout, no down/upsampling, last layer zero-initialised). Input channels: current SLF, distance map, ray-cast shadowing, building mask, free-space anchor — five physically distinct fields per emitter. Output: a residual to the free-space anchor.
Everything else is closed-form and parameter-free. Compared to your current model, you trade two ConditionedProximalBlocks per iteration for one ray-cast-conditioned VkV_k
Vk​ — and you delete the three mode-SVT blocks. Net parameter count should go *down*, not up.
7. Initialization and a few practical notes
Number of emitters RR
R. Count connected components in the TXPOS map. If the map is fuzzy (heatmap rather than mask), threshold + peak-find. In your BART-Lab setup each cropped 256×256 has one base station, so R=1R = 1
R=1 initially — but the framework handles R>1R > 1
R>1 for SpectrumNet's dense-urban scenes where multiple BSs cover a tile.
SLF and PSD initialisation. Set Sr(0)(x)=Prfree(x)\mathbf{S}_r^{(0)}(x) = \mathbf{P}_r^{\text{free}}(x)
Sr(0)​(x)=Prfree​(x) — the physics-only anchor. Set \mathbf{c}_r^{(0)}(k) = $ least-squares fit of observed $\tilde{\mathcal{D}}
 to ∑rSr(0)cr\sum_r \mathbf{S}_r^{(0)} c_r
∑r​Sr(0)​cr​ restricted to Ω\Omega
Ω. This gives the network a sensible starting point and dramatically shortens training compared to all-zeros init.
When RR
R is wrong / unknown. Set RR
R to a safe upper bound (e.g., 4). The non-negativity constraint on cr\mathbf{c}_r
cr​ plus the learned proximal will drive unused emitters' PSDs toward zero — same logic as over-complete dictionary learning. You can add a small ∑r∥cr∥1\sum_r \|\mathbf{c}_r\|_1
∑r​∥cr​∥1​ penalty to encourage this explicitly.
Loss function. Keep the L1 reconstruction loss on D^\hat{\mathcal{D}}
D^ vs ground truth. Optionally add the shadowing-statistic auxiliary loss from RadioDUN: encourage D^−Prfree\hat{\mathcal{D}} - \mathbf{P}_r^{\text{free}}
D^−Prfree​ to be zero-mean Gaussian per emitter. Cheap, and gives you an interpretable training signal that aligns with the standard log-normal shadowing model.
Identifiability check. The BTD model has well-known identifiability theorems (Zhang–Fu–Wang–Zhao–Hong, IEEE TSP 2020): the {Sr,cr}\{\mathbf{S}_r, \mathbf{c}_r\}
{Sr​,cr​} are recoverable up to scaling/permutation provided the cr\mathbf{c}_r
cr​ are pairwise non-collinear and the SLFs are linearly independent. These are mild and almost always satisfied — but it's worth a sentence in your paper, because reviewers from the signal-processing side will appreciate that your decomposition has provable structure where the original "low-rank background" did not.

Two things you'll want to think about for the next step: (1) the data loader needs to compute Tr\mathcal{T}_r
Tr​ and drd_r
dr​ once per sample at load time and cache them — these are deterministic given (TX_position, building_map), so they belong next to your existing env_map loading code; (2) the model file gets *simpler*, not more complex, because you delete `calMiBlock` entirely and replace `ConditionedProximalBlock` with a residual CNN that takes the physical fields. Happy to write the corresponding `model.py` and `util.py` rewrites whenever you want to move to implementation.