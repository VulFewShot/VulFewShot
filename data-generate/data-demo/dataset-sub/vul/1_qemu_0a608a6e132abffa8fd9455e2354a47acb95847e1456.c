int FUN1(VAR1 *VAR2, uint8_t VAR3, uint64_t VAR4, uint8_t VAR5)
{
    VAR6 *VAR7 = &VAR2->VAR7;
    uint32_t VAR8;
    ZpciFib VAR9;
    VAR10 *VAR11;
    uint32_t VAR12;
    uint64_t VAR13 = VAR14;
    if (VAR7->VAR15.VAR16 & VAR17) {
        FUN2(VAR7, VAR18, 6);
        return 0;
    }
    VAR8 = VAR7->VAR19[VAR3] >> 32;
    if (VAR4 & 0x7) {
        FUN2(VAR7, VAR20, 6);
        return 0;
    }
    VAR11 = FUN3(VAR8);
    if (!VAR11) {
        FUN4(VAR2, VAR21);
        return 0;
    }
    memset(&VAR9, 0, sizeof(VAR9));
    switch (VAR11->VAR22) {
    case VAR23:
    case VAR24:
        FUN4(VAR2, VAR21);
        return 0;
    case VAR25:
        if (VAR8 & VAR26) {
            FUN4(VAR2, VAR21);
            return 0;
        }
        goto VAR27;
    
    case VAR28:
        VAR9.VAR29 |= 0x20;
    case VAR30:
        VAR9.VAR29 |= 0x40;
    case VAR31:
        VAR9.VAR29 |= 0x80;
        if (VAR11->VAR32) {
            VAR9.VAR29 |= 0x10;
        }
        if (!(VAR8 & VAR26)) {
            VAR7->VAR19[VAR3] |= 1ULL << 63;
        }
        break;
    case VAR33:
        FUN4(VAR2, VAR34);
        FUN5(VAR7, VAR3, VAR35);
        return 0;
    }
    FUN6(&VAR9.VAR36, VAR11->VAR36);
    FUN6(&VAR9.VAR37, VAR11->VAR37);
    FUN6(&VAR9.VAR38, VAR11->VAR39);
    FUN6(&VAR9.VAR40, VAR11->VAR41.VAR42.VAR43);
    FUN6(&VAR9.VAR44, VAR11->VAR41.VAR42.VAR45);
    FUN6(&VAR9.VAR46, VAR11->VAR46);
    VAR12 = ((VAR47)VAR11->VAR48 << 28) | ((VAR47)VAR11->VAR49 << 16) |
           ((VAR47)VAR11->VAR41.VAR42.VAR50 << 8) |
           ((VAR47)VAR11->VAR51 << 7) | VAR11->VAR41.VAR42.VAR52;
    FUN7(&VAR9.VAR12, VAR12);
VAR27:
    if (FUN8(VAR2, VAR4, VAR5, (VAR53 *)&VAR9, sizeof(VAR9))) {
        return 0;
    }
    FUN4(VAR2, VAR13);
    return 0;
}