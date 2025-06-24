---
title: "UART_ECHOBACK_LAB"
thumbnail: "/assets/img/thumbnail/arm_stm32.jpeg"
---

# Code
---
```
#include "device_driver.h"

static void Sys_Init(void)
{
	Clock_Init();
	LED_Init();
	Uart_Init(115200);
	Key_Poll_Init();
}

// 받은 글자를 다시 UART로 출력
void Main(void)
{
	Sys_Init();
	Uart_Printf("UART Echo-Back Test\n");

	for(;;)
	{
		unsigned char x;

		while(!Macro_Check_Bit_Set(USART1->SR, 5));
		x = USART1->DR;
		while(!Macro_Check_Bit_Set(USART1->SR, 7));
		USART1->DR = x;
	}
}
```

# 분석
---