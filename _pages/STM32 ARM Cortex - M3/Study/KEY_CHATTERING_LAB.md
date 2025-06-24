---
title: "KEY_CHATTERING_LAB"
thumbnail: "/assets/img/thumbnail/arm_stm32.jpeg"
---

# Code
---
```
#include "device_driver.h"

void Key_Poll_Init(void)
{
	Macro_Set_Bit(RCC->APB2ENR, 3);
	Macro_Write_Block(GPIOB->CRL, 0xff, 0x44, 24);
}

/* 0: 디버깅용 설정 */
/* 1: 정상 동작용 설정 */

#if 0
	#define N 		20000
#else
	#define N 		3000000
#endif

static int Key_Check_Input(void)
{
	return Macro_Extract_Area(~GPIOB->IDR, 0x3, 6);
}

// N 만큼 같은 값이 읽혀야 Key 값으로 판정
int Key_Get_Pressed(void)
{
	unsigned int i, k;

	for(;;)
	{
		k = Key_Check_Input();

		for(i=0; i<N; i++)
		{
			if(k != Key_Check_Input())
			{
				break;
			}
		}

		if(i == N) return k;;
	}
}

void Key_Wait_Key_Released(void)
{
	while(Key_Get_Pressed());
}

int Key_Wait_Key_Pressed(void)
{
	int k;

	while((k = Key_Get_Pressed()) == 0);
	return k;
}
```

# 분석
---