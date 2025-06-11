import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show


def process_remote_sensing_image(image_path, output_path):
    """
    处理遥感图像数据：
    1. 使用rasterio读取包含5个波段（RGB、近红外、短红外）的遥感图像
    2. 提取RGB三个可见光波段
    3. 将数值范围从0-10000压缩到0-255
    4. 转换为8位RGB图像并保存
    5. 使用matplotlib显示原始图像和处理后的图像

    参数:
    image_path -- 输入遥感图像路径
    output_path -- 输出RGB图像路径
    """
    # 使用rasterio读取遥感图像
    with rasterio.open(image_path) as src:
        # 读取所有波段数据 (形状: [波段数, 高度, 宽度])
        data = src.read()

        # 获取元数据
        profile = src.profile
        print(f"图像尺寸: {src.height} x {src.width}")
        print(f"波段数: {src.count}")
        print(f"数据类型: {src.dtypes[0]}")

        # 转置为(高度, 宽度, 波段)格式
        image_data = data.transpose(1, 2, 0).astype(np.float32)

        # 可视化原始图像（使用rasterio内置函数）
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        show(src, title='原始遥感图像')

        # 提取RGB波段 (假设前三个波段是红、绿、蓝)
        # 注意：不同传感器波段顺序可能不同，请根据实际情况调整
        rgb_bands = image_data[:, :, :3].copy()

        # 打印原始数据范围
        print("\n原始数据范围:")
        for i, band in enumerate(['红', '绿', '蓝']):
            print(f"{band}波段: 最小值={np.min(rgb_bands[:, :, i]):.2f}, 最大值={np.max(rgb_bands[:, :, i]):.2f}")

        # 方法1: 线性缩放 (简单快速)
        # rgb_normalized = (rgb_bands / 10000.0) * 255.0

        # 方法2: 百分比截断拉伸 (增强对比度，推荐)
        # 计算2%和98%分位数，排除极端值
        lower_percentile = np.percentile(rgb_bands, 2, axis=(0, 1))
        upper_percentile = np.percentile(rgb_bands, 98, axis=(0, 1))
        print("\n拉伸范围 (2%-98%分位数):")
        for i, band in enumerate(['红', '绿', '蓝']):
            print(f"{band}波段: {lower_percentile[i]:.2f} - {upper_percentile[i]:.2f}")

        # 应用线性拉伸
        rgb_normalized = np.zeros_like(rgb_bands)
        for i in range(3):
            # 防止除以零
            band_range = upper_percentile[i] - lower_percentile[i]
            if band_range <= 0:
                band_range = 1
            # 拉伸并缩放到0-255
            rgb_normalized[:, :, i] = np.clip(
                (rgb_bands[:, :, i] - lower_percentile[i]) / band_range * 255,
                0, 255
            )

        # 转换为8位整数类型
        rgb_8bit = rgb_normalized.astype(np.uint8)

        # 使用OpenCV保存图像 (注意OpenCV使用BGR顺序)
        # 方法1: 直接保存RGB (但OpenCV会误以为是BGR)
        # cv2.imwrite(output_path, rgb_8bit)

        # 方法2: 转换为BGR再保存 (推荐)
        bgr_image = cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr_image)

        # 使用matplotlib显示处理后的图像
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_8bit)
        plt.title('处理后的RGB图像')
        plt.axis('off')

        # 保存对比图
        plt.tight_layout()
        plt.savefig('comparison.png', dpi=150)
        plt.show()

        print(f"\n处理完成! 结果已保存至: {output_path}")


# 示例用法
if __name__ == "__main__":
    # 输入和输出文件路径
    input_image = r"C:\00Code\python-code\PythonProject\2019_1101_nofire_B2348_B12_10m_roi.tif"  # 替换为你的遥感图像路径
    output_image = "output_rgb.jpg"

    # 处理图像
    process_remote_sensing_image(input_image, output_image)