---
title: "KEY_IN_LAB"
thumbnail: "/assets/img/thumbnail/arm_stm32.jpeg"
---

# Code
---
**LAB1 : KEY 인식**
```
/* Key 인식 #1 */

#if 1

void Main(void)
{
	int value = 0;

	Sys_Init();
	Uart_Printf("KEY Input Test #1\n");


	Macro_Write_Block(GPIOB->CRL, 0xff, 0x44, 24);

	for(;;)
	{
		if(Macro_Check_Bit_Clear(GPIOB->IDR, 6)) Macro_Set_Bit(value, 0);
		else Macro_Clear_Bit(value, 0);

		if(Macro_Check_Bit_Clear(GPIOB->IDR, 7)) Macro_Set_Bit(value, 1);
		else Macro_Clear_Bit(value, 1);

		LED_Display(value);
	}
}

#endif

/* Extract Macro를 이용한 Key 인식 #2 */

#if 0

void Main(void)
{
	Sys_Init();
	Uart_Printf("KEY Input Test #2\n");

	Macro_Write_Block(GPIOB->CRL, 0xff, 0x44, 24);

	for(;;)
	{
		Macro_Write_Block(GPIOB->ODR,0x3,Macro_Extract_Area(GPIOB->IDR,0x3,6),8);
	}
}

#endif
```
**LAB2 : KEY에 의한 LED Toggling**
```
/* Key에 의한 LED Toggling */

#if 1

void Main(void)
{
	Sys_Init();
	Uart_Printf("KEY Input Toggling #1\n");

	Macro_Write_Block(GPIOB->CRL, 0xff, 0x44, 24);

	for(;;)
	{
		if(Macro_Check_Bit_Clear(GPIOB->IDR, 6))
		{
			Macro_Invert_Bit(GPIOB->ODR, 8);
		}
	}
}

#endif

/* Key Released 상태 대기에 의한 LED Toggling */

#if 0

void Main(void)
{
	Sys_Init();
	Uart_Printf("KEY Input Toggling #2\n");

	Macro_Write_Block(GPIOB->CRL, 0xff, 0x44, 24);

	for(;;)
	{
		if(Macro_Check_Bit_Clear(GPIOB->IDR, 6))
		{
			Macro_Invert_Bit(GPIOB->ODR, 8);
			while(!Macro_Check_Bit_Set(GPIOB->IDR, 6));
		}
	}
}

#endif

/* Inter-Lock을 적용한 Key에 의한 LED Toggling */

#if 0

void Main(void)
{
	int interlock = 1;

	Sys_Init();
	Uart_Printf("KEY Input Toggling #3\n");

	Macro_Write_Block(GPIOB->CRL, 0xff, 0x44, 24);

	for(;;)
	{
		if((interlock != 0) && Macro_Check_Bit_Clear(GPIOB->IDR, 6))
		{
			Macro_Invert_Bit(GPIOB->ODR, 8);
			interlock = 0;
		}

		else if((interlock == 0) && Macro_Check_Bit_Set(GPIOB->IDR, 6))
		{
			interlock = 1;
		}
	}
}

#endif
```

# 분석
---