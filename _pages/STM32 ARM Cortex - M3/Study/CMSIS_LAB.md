---
title: "CMSIS_LAB"
thumbnail: "/assets/img/thumbnail/arm_stm32.jpeg"
---

# Code
---
```
#include "device_driver.h"
 
void Main(void)
{
    volatile int i;
 
    Uart_Init(115200);
    Uart_Printf("CMSIS Based Register Define\n");
 
    RCC->APB2ENR |= (1<<3);
 
    // LED Pin을 출력으로 설정
    GPIOB->CRH = 0x66 << 0;
 
    for(;;)
    {
        // LED 모두 ON
        GPIOB->ODR = 0x0 << 8;
        for(i=0; i<0x40000; i++);

        // LED 모두 OFF
        GPIOB->ODR = 0x3 << 8;
        for(i=0; i<0x40000; i++);
    }
}
```

# 분석
---