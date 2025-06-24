---
title: "KEY_DRIVER_LAB"
thumbnail: "/assets/img/thumbnail/arm_stm32.jpeg"
---

# Code
---
```
#define KEY0_PUSH() 	(Macro_Check_Bit_Clear(GPIOB->IDR, 6))
#define KEY0_REL() 		(Macro_Check_Bit_Set(GPIOB->IDR, 6))
#define KEY1_PUSH() 	(Macro_Check_Bit_Clear(GPIOB->IDR, 7))
#define KEY1_REL() 		(Macro_Check_Bit_Set(GPIOB->IDR, 7))

#define KEY_VALUE()		(Macro_Extract_Area(~GPIOB->IDR, 0x3, 6))

int Key_Get_Pressed(void)
{
	#if 0
	if( KEY0_REL() &&  KEY1_REL() ) return 0;
	if( KEY0_PUSH() && KEY1_REL() ) return 1;
	if( KEY0_REL() && KEY1_PUSH() ) return 2;
	if( KEY0_PUSH() && KEY1_PUSH() ) return 3;
	#endif

	return KEY_VALUE();
}

void Key_Wait_Key_Released(void)
{
	#if 0
	for(;;)
	{
		if( KEY_VALUE() == 0 ) return;
	}
	#endif

	while( !(KEY_VALUE() == 0) );
}

int Key_Wait_Key_Pressed(void)
{
	#if 0
	for(;;)
	{
		int key = KEY_VALUE();
		if( key != 0 ) return;
	}
	#endif

	while( !((key = KEY_VALUE()) != 0)) );
	return key;
}

```

# 분석
---