def test_monai_installation():
    try:
        # 测试cuda
        import torch
        print(f"✓ CUDA version: {torch.cuda.get_device_name(0)}")
        
        import monai
        print(f"✓ MONAI version: {monai.__version__}")
        
        from monai.config import print_config
        print("\n详细配置:")
        print_config()
        
        # 简单功能测试
        from monai.transforms import RandRotate90
        import torch
        transform = RandRotate90()
        data = torch.rand(1, 32, 32, 32)
        _ = transform(data)
        print("\n✓ 基础变换测试通过")
        
        print("\n✓ MONAI 安装成功!")
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

test_monai_installation()
