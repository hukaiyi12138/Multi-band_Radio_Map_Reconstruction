# 改动
把CNN改成UNet网络，避开CNN局限的感受野，使用UNet扩充至全局视野
目前在fiber1条件下效果超越RadioUNet，并且fiber整体除了fiber10都有提升
但问题是mask所有比CNN效果差，从可视化图来看是UNet把采样点做成了噪点
目前UNet + mu_grad=0.15效果不好，现在尝试去掉mu_grad=0.15，改成mu_grad=0，只用L1损失

跑完 mu_grad=0.0 后：
    如果可视化图里噪点没了 → 梯度损失就是元凶，论文用 mu_grad=0.0 重跑
    如果可视化图里噪点还在，只是淡了一些 → U-Net 本身也是问题，需要更深入调整
    如果可视化图里噪点还是很明显 → 必须回退到 CNN

HINT - 目前是try3版本 2026.06.20 21h53m
