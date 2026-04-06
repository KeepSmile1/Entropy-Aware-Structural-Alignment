#!/bin/bash

# 设置参数范围
start=40
end=10
step=10

# 中断清理函数
cleanup() {
    echo "中断信号捕获，正在清理资源..."
    cp config_backup.py config.py
    exit 1
}
trap cleanup SIGINT SIGTERM

# 备份原始配置文件
cp config.py config_backup.py

# 遍历参数 m 从 start 到 end，步长为负 step
for (( m=$start; m>=$end; m-=$step ))
do
    echo "当前实验参数: m=$m"

    sed -i "s/^m=.*/m=$m/" config.py

    # 打印修改后的 m 值
    echo "修改后的 m 参数: $(grep '^m=' config.py)"
    # 运行训练任务，直到成功为止
    until torchrun --nproc_per_node=1 train.py
    do
        echo "实验 m=$m 失败，5 秒后重试..."
        sleep 5
    done

    echo "实验 m=$m 成功完成 ✅"
    echo "============================"
done

# 恢复原始配置文件
cp config_backup.py config.py
echo "所有实验完成 🎉"