---
title: "LED_ON_LAB"
thumbnail: "/assets/img/thumbnail/arm_stm32.jpeg"
---

# Code
---
```
#define RCC_APB2ENR (*(unsigned long*)0x40021018)
 
#define GPIOB_CRH   (*(unsigned long*)0x40010C04)
#define GPIOB_ODR   (*(unsigned long*)0x40010C0C)
 
void Main(void)
{
    RCC_APB2ENR |= (1<<3);
 
    GPIOB_CRH = 0x66 << 0;
    GPIOB_ODR = 0x01 << 8;
}
```

# 분석
---