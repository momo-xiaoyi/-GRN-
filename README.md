# -GRN-
基因调控网络实现群机器人自组织的代码。
这方法主要是通过模拟基因表达结构，构造群机器人运动方程（微分方程/递归方程）。可以理解为人工势场法的变形（在传统人工势场法中增加了非线性结构）。
以上代码，实现，20个机器人形成单位圆。机器人数量和障碍物数量在代码里可以调节。
代码结果可能会出现没那么圆的结构情况，在论文中主要靠二分法，优化排斥力的作用范围，来使结果更好。
