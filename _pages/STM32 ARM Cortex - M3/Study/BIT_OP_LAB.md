---
title: "BIT_OP_LAB"
thumbnail: "/assets/img/thumbnail/arm_stm32.jpeg"
---

# Code
---
**LAB1 : 비트 연산에 의한 LED ON**
```
#if 1
 
void Main(void)
{
    RCC->APB2ENR |= (1<<3);
 
    GPIOB->CRH &=~((1<<7)|(3<<3)|(1<<0));
    GPIOB->CRH |= (3<<5)|(3<<1);
 
    GPIOB->ODR &=~(1<<8);
    GPIOB->ODR |= (1<<9);
}
 
#endif
 
#if 0
 
void Main(void)
{
    RCC->APB2ENR |= (1<<3);
 
    GPIOB->CRH = (GPIOB->CRH & ~((1<<7)|(3<<3)|(1<<0)))|((3<<5)|(3<<1));
    GPIOB->ODR = (GPIOB->ODR & ~(0x1<<8)) | (1<<9);
}
 
#endif
 
#if 0
 
void Main(void)
{
    RCC->APB2ENR |= (1<<3);
 
    // LED0을 ON, LED을 OFF
    GPIOB->CRH = (GPIOB->CRH & ~(0xff<<0)) | (0x66<<0);
    GPIOB->ODR = (GPIOB->ODR & ~(0x3<<8)) | (0x1<<8);
}
 
#endif
```
**LAB2 : 비트 연산 Macro 활용에 의한 LED ON**
```
void Main(void)
{
    Macro_Set_Bit(RCC->APB2ENR, 3);
 
    // LED0을 ON, LED을 OFF
    Macro_Write_Block(GPIOB->CRH, 0xff, 0x66, 0);
    Macro_Write_Block(GPIOB->ODR, 0x3, 0x2, 8);
}
```
**LAB3 : 비트 연산 Macro 활용에 의한 LED Toggling**
```
void Main(void)
{
    volatile int i;
 
    Macro_Set_Bit(RCC->APB2ENR, 3);
 
    // 초기에 LED 모두 OFF
    Macro_Write_Block(GPIOB->CRH, 0xff, 0x66, 0);
    Macro_Set_Area(GPIOB->ODR, 0x3, 8);
 
    // LED 반전 및 Delay 설정
    for(;;)
    {
        Macro_Invert_Area(GPIOB->ODR, 0x3, 8);
        for(i=0; i<0x80000; i++);
    }
}
```

# 분석
---