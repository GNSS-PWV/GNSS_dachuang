2026.2.6: 
使用更简单的attention；
增加了dropout，添加了 LayerNorm （在注意力机制前添加层归一化）
改用全局平均池化 ：使用 torch.mean(attn_out, dim=1) 替代仅使用最后一个时间步的输出
结果：CNN+bilstm+attention的r2从0.64上升到0.67-0.69
