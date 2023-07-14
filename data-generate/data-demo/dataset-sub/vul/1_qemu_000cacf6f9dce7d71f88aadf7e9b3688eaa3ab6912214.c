static void FUN1(VAR1 *VAR2, int VAR3, target_ulong VAR4, int VAR5)
{
    int VAR6, VAR7, VAR8, VAR9, VAR10, VAR11;
    int VAR12, VAR13, VAR14, VAR15, VAR16, VAR17;
    void *VAR18;
    VAR3 &= 0xff;
    if (VAR2->VAR19 & VAR20)
        VAR6 = 1;
    else if (VAR2->VAR19 & VAR21)
        VAR6 = 2;
    else if (VAR2->VAR19 & VAR22)
        VAR6 = 3;
    else
        VAR6 = 0;
    VAR18 = VAR23[VAR3][VAR6];
    if (!VAR18)
        goto VAR24;
    if ((VAR3 <= 0x5f && VAR3 >= 0x10) || VAR3 == 0xc6 || VAR3 == 0xc2) {
        VAR9 = 1;
    } else {
        if (VAR6 == 0) {
            
            VAR9 = 0;
        } else {
            VAR9 = 1;
        }
    }
    
    if (VAR2->VAR25 & VAR26) {
        FUN2(VAR2, VAR27, VAR4 - VAR2->VAR28);
        return;
    }
    if (VAR2->VAR25 & VAR29) {
    VAR24:
        FUN2(VAR2, VAR30, VAR4 - VAR2->VAR28);
        return;
    }
    if (VAR9 && !(VAR2->VAR25 & VAR31))
        if ((VAR3 != 0x38 && VAR3 != 0x3a) || (VAR2->VAR19 & VAR20))
            goto VAR24;
    if (VAR3 == 0x0e) {
        if (!(VAR2->VAR32 & VAR33))
            goto VAR24;
        
        FUN3(VAR34);
        return;
    }
    if (VAR3 == 0x77) {
        
        FUN3(VAR34);
        return;
    }
    
    if (!VAR9) {
        FUN3(VAR35);
    }
    VAR12 = FUN4(VAR2->VAR36++);
    VAR15 = ((VAR12 >> 3) & 7);
    if (VAR9)
        VAR15 |= VAR5;
    VAR13 = (VAR12 >> 6) & 3;
    if (VAR18 == VAR37) {
        VAR3 |= (VAR6 << 8);
        switch(VAR3) {
        case 0x0e7: 
            if (VAR13 == 3)
                goto VAR24;
            FUN5(VAR2, VAR12, &VAR16, &VAR17);
            FUN6(VAR2->VAR38, FUN7(VAR39,VAR40[VAR15].VAR41));
            break;
        case 0x1e7: 
        case 0x02b: 
        case 0x12b: 
        case 0x3f0: 
            if (VAR13 == 3)
                goto VAR24;
            FUN5(VAR2, VAR12, &VAR16, &VAR17);
            FUN8(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15]));
            break;
        case 0x6e: 
#ifdef VAR43
            if (VAR2->VAR44 == 2) {
                FUN9(VAR2, VAR12, VAR45, VAR46, 0);
                FUN10(VAR47[0], VAR48, FUN7(VAR39,VAR40[VAR15].VAR41));
            } else
#endif
            {
                FUN9(VAR2, VAR12, VAR49, VAR46, 0);
                FUN11(VAR50, VAR48, 
                                 FUN7(VAR39,VAR40[VAR15].VAR41));
                FUN12(VAR51, VAR50, VAR47[0]);
            }
            break;
        case 0x16e: 
#ifdef VAR43
            if (VAR2->VAR44 == 2) {
                FUN9(VAR2, VAR12, VAR45, VAR46, 0);
                FUN11(VAR50, VAR48, 
                                 FUN7(VAR39,VAR42[VAR15]));
                FUN12(VAR52, VAR50, VAR47[0]);
            } else
#endif
            {
                FUN9(VAR2, VAR12, VAR49, VAR46, 0);
                FUN11(VAR50, VAR48, 
                                 FUN7(VAR39,VAR42[VAR15]));
                FUN13(VAR53, VAR47[0]);
                FUN12(VAR54, VAR50, VAR53);
            }
            break;
        case 0x6f: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN14(VAR2->VAR38, FUN7(VAR39,VAR40[VAR15].VAR41));
            } else {
                VAR14 = (VAR12 & 7);
                FUN15(VAR55, VAR48,
                               FUN7(VAR39,VAR40[VAR14].VAR41));
                FUN16(VAR55, VAR48,
                               FUN7(VAR39,VAR40[VAR15].VAR41));
            }
            break;
        case 0x010: 
        case 0x110: 
        case 0x028: 
        case 0x128: 
        case 0x16f: 
        case 0x26f: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN17(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15]));
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN19(FUN7(VAR39,VAR42[VAR15]),
                            FUN7(VAR39,VAR42[VAR14]));
            }
            break;
        case 0x210: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN20(VAR49 + VAR2->VAR38);
                FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR42[VAR15].FUN22(0)));
                FUN23();
                FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR42[VAR15].FUN22(1)));
                FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR42[VAR15].FUN22(2)));
                FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR42[VAR15].FUN22(3)));
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN24(FUN7(VAR39,VAR42[VAR15].FUN22(0)),
                            FUN7(VAR39,VAR42[VAR14].FUN22(0)));
            }
            break;
        case 0x310: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN14(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15].FUN25(0)));
                FUN23();
                FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR42[VAR15].FUN22(2)));
                FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR42[VAR15].FUN22(3)));
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN26(FUN7(VAR39,VAR42[VAR15].FUN25(0)),
                            FUN7(VAR39,VAR42[VAR14].FUN25(0)));
            }
            break;
        case 0x012: 
        case 0x112: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN14(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15].FUN25(0)));
            } else {
                
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN26(FUN7(VAR39,VAR42[VAR15].FUN25(0)),
                            FUN7(VAR39,VAR42[VAR14].FUN25(1)));
            }
            break;
        case 0x212: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN17(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15]));
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN24(FUN7(VAR39,VAR42[VAR15].FUN22(0)),
                            FUN7(VAR39,VAR42[VAR14].FUN22(0)));
                FUN24(FUN7(VAR39,VAR42[VAR15].FUN22(2)),
                            FUN7(VAR39,VAR42[VAR14].FUN22(2)));
            }
            FUN24(FUN7(VAR39,VAR42[VAR15].FUN22(1)),
                        FUN7(VAR39,VAR42[VAR15].FUN22(0)));
            FUN24(FUN7(VAR39,VAR42[VAR15].FUN22(3)),
                        FUN7(VAR39,VAR42[VAR15].FUN22(2)));
            break;
        case 0x312: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN14(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15].FUN25(0)));
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN26(FUN7(VAR39,VAR42[VAR15].FUN25(0)),
                            FUN7(VAR39,VAR42[VAR14].FUN25(0)));
            }
            FUN26(FUN7(VAR39,VAR42[VAR15].FUN25(1)),
                        FUN7(VAR39,VAR42[VAR15].FUN25(0)));
            break;
        case 0x016: 
        case 0x116: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN14(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15].FUN25(1)));
            } else {
                
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN26(FUN7(VAR39,VAR42[VAR15].FUN25(1)),
                            FUN7(VAR39,VAR42[VAR14].FUN25(0)));
            }
            break;
        case 0x216: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN17(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15]));
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN24(FUN7(VAR39,VAR42[VAR15].FUN22(1)),
                            FUN7(VAR39,VAR42[VAR14].FUN22(1)));
                FUN24(FUN7(VAR39,VAR42[VAR15].FUN22(3)),
                            FUN7(VAR39,VAR42[VAR14].FUN22(3)));
            }
            FUN24(FUN7(VAR39,VAR42[VAR15].FUN22(0)),
                        FUN7(VAR39,VAR42[VAR15].FUN22(1)));
            FUN24(FUN7(VAR39,VAR42[VAR15].FUN22(2)),
                        FUN7(VAR39,VAR42[VAR15].FUN22(3)));
            break;
        case 0x7e: 
#ifdef VAR43
            if (VAR2->VAR44 == 2) {
                FUN15(VAR47[0], VAR48, 
                               FUN7(VAR39,VAR40[VAR15].VAR41));
                FUN9(VAR2, VAR12, VAR45, VAR46, 1);
            } else
#endif
            {
                FUN27(VAR47[0], VAR48, 
                                 FUN7(VAR39,VAR40[VAR15].VAR41.FUN28(0)));
                FUN9(VAR2, VAR12, VAR49, VAR46, 1);
            }
            break;
        case 0x17e: 
#ifdef VAR43
            if (VAR2->VAR44 == 2) {
                FUN15(VAR47[0], VAR48, 
                               FUN7(VAR39,VAR42[VAR15].FUN25(0)));
                FUN9(VAR2, VAR12, VAR45, VAR46, 1);
            } else
#endif
            {
                FUN27(VAR47[0], VAR48, 
                                 FUN7(VAR39,VAR42[VAR15].FUN22(0)));
                FUN9(VAR2, VAR12, VAR49, VAR46, 1);
            }
            break;
        case 0x27e: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN14(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15].FUN25(0)));
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN26(FUN7(VAR39,VAR42[VAR15].FUN25(0)),
                            FUN7(VAR39,VAR42[VAR14].FUN25(0)));
            }
            FUN29(FUN7(VAR39,VAR42[VAR15].FUN25(1)));
            break;
        case 0x7f: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN6(VAR2->VAR38, FUN7(VAR39,VAR40[VAR15].VAR41));
            } else {
                VAR14 = (VAR12 & 7);
                FUN26(FUN7(VAR39,VAR40[VAR14].VAR41),
                            FUN7(VAR39,VAR40[VAR15].VAR41));
            }
            break;
        case 0x011: 
        case 0x111: 
        case 0x029: 
        case 0x129: 
        case 0x17f: 
        case 0x27f: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN8(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15]));
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN19(FUN7(VAR39,VAR42[VAR14]),
                            FUN7(VAR39,VAR42[VAR15]));
            }
            break;
        case 0x211: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN27(VAR47[0], VAR48, FUN7(VAR39,VAR42[VAR15].FUN22(0)));
                FUN30(VAR49 + VAR2->VAR38);
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN24(FUN7(VAR39,VAR42[VAR14].FUN22(0)),
                            FUN7(VAR39,VAR42[VAR15].FUN22(0)));
            }
            break;
        case 0x311: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN6(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15].FUN25(0)));
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN26(FUN7(VAR39,VAR42[VAR14].FUN25(0)),
                            FUN7(VAR39,VAR42[VAR15].FUN25(0)));
            }
            break;
        case 0x013: 
        case 0x113: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN6(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15].FUN25(0)));
            } else {
                goto VAR24;
            }
            break;
        case 0x017: 
        case 0x117: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN6(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15].FUN25(1)));
            } else {
                goto VAR24;
            }
            break;
        case 0x71: 
        case 0x72:
        case 0x73:
        case 0x171: 
        case 0x172:
        case 0x173:
            VAR10 = FUN4(VAR2->VAR36++);
            if (VAR9) {
                FUN31(VAR10);
                FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR56.FUN22(0)));
                FUN23();
                FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR56.FUN22(1)));
                VAR7 = FUN7(VAR39,VAR56);
            } else {
                FUN31(VAR10);
                FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR57.FUN28(0)));
                FUN23();
                FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR57.FUN28(1)));
                VAR7 = FUN7(VAR39,VAR57);
            }
            VAR18 = VAR58[((VAR3 - 1) & 3) * 8 + (((VAR12 >> 3)) & 7)][VAR6];
            if (!VAR18)
                goto VAR24;
            if (VAR9) {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                VAR8 = FUN7(VAR39,VAR42[VAR14]);
            } else {
                VAR14 = (VAR12 & 7);
                VAR8 = FUN7(VAR39,VAR40[VAR14].VAR41);
            }
            FUN11(VAR50, VAR48, VAR8);
            FUN11(VAR59, VAR48, VAR7);
            FUN12(VAR18, VAR50, VAR59);
            break;
        case 0x050: 
            VAR14 = (VAR12 & 7) | FUN18(VAR2);
            FUN11(VAR50, VAR48, 
                             FUN7(VAR39,VAR42[VAR14]));
            FUN32(VAR60, VAR53, VAR50);
            FUN33(VAR47[0], VAR53);
            FUN34(VAR49, VAR15);
            break;
        case 0x150: 
            VAR14 = (VAR12 & 7) | FUN18(VAR2);
            FUN11(VAR50, VAR48, 
                             FUN7(VAR39,VAR42[VAR14]));
            FUN32(VAR61, VAR53, VAR50);
            FUN33(VAR47[0], VAR53);
            FUN34(VAR49, VAR15);
            break;
        case 0x02a: 
        case 0x12a: 
            FUN3(VAR35);
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                VAR8 = FUN7(VAR39,VAR57);
                FUN14(VAR2->VAR38, VAR8);
            } else {
                VAR14 = (VAR12 & 7);
                VAR8 = FUN7(VAR39,VAR40[VAR14].VAR41);
            }
            VAR7 = FUN7(VAR39,VAR42[VAR15]);
            FUN11(VAR50, VAR48, VAR7);
            FUN11(VAR59, VAR48, VAR8);
            switch(VAR3 >> 8) {
            case 0x0:
                FUN12(VAR62, VAR50, VAR59);
                break;
            default:
            case 0x1:
                FUN12(VAR63, VAR50, VAR59);
                break;
            }
            break;
        case 0x22a: 
        case 0x32a: 
            VAR11 = (VAR2->VAR44 == 2) ? VAR45 : VAR49;
            FUN9(VAR2, VAR12, VAR11, VAR46, 0);
            VAR7 = FUN7(VAR39,VAR42[VAR15]);
            FUN11(VAR50, VAR48, VAR7);
            VAR18 = VAR64[(VAR2->VAR44 == 2) * 2 + ((VAR3 >> 8) - 2)];
            if (VAR11 == VAR49) {
                FUN13(VAR53, VAR47[0]);
                FUN12(VAR18, VAR50, VAR53);
            } else {
                FUN12(VAR18, VAR50, VAR47[0]);
            }
            break;
        case 0x02c: 
        case 0x12c: 
        case 0x02d: 
        case 0x12d: 
            FUN3(VAR35);
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                VAR8 = FUN7(VAR39,VAR56);
                FUN17(VAR2->VAR38, VAR8);
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                VAR8 = FUN7(VAR39,VAR42[VAR14]);
            }
            VAR7 = FUN7(VAR39,VAR40[VAR15 & 7].VAR41);
            FUN11(VAR50, VAR48, VAR7);
            FUN11(VAR59, VAR48, VAR8);
            switch(VAR3) {
            case 0x02c:
                FUN12(VAR65, VAR50, VAR59);
                break;
            case 0x12c:
                FUN12(VAR66, VAR50, VAR59);
                break;
            case 0x02d:
                FUN12(VAR67, VAR50, VAR59);
                break;
            case 0x12d:
                FUN12(VAR68, VAR50, VAR59);
                break;
            }
            break;
        case 0x22c: 
        case 0x32c: 
        case 0x22d: 
        case 0x32d: 
            VAR11 = (VAR2->VAR44 == 2) ? VAR45 : VAR49;
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                if ((VAR3 >> 8) & 1) {
                    FUN14(VAR2->VAR38, FUN7(VAR39,VAR56.FUN25(0)));
                } else {
                    FUN20(VAR49 + VAR2->VAR38);
                    FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR56.FUN22(0)));
                }
                VAR8 = FUN7(VAR39,VAR56);
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                VAR8 = FUN7(VAR39,VAR42[VAR14]);
            }
            VAR18 = VAR64[(VAR2->VAR44 == 2) * 2 + ((VAR3 >> 8) - 2) + 4 +
                                    (VAR3 & 1) * 4];
            FUN11(VAR50, VAR48, VAR8);
            if (VAR11 == VAR49) {
                FUN32(VAR18, VAR53, VAR50);
                FUN33(VAR47[0], VAR53);
            } else {
                FUN32(VAR18, VAR47[0], VAR50);
            }
            FUN34(VAR11, VAR15);
            break;
        case 0xc4: 
        case 0x1c4:
            VAR2->VAR69 = 1;
            FUN9(VAR2, VAR12, VAR70, VAR46, 0);
            VAR10 = FUN4(VAR2->VAR36++);
            if (VAR6) {
                VAR10 &= 7;
                FUN35(VAR47[0], VAR48,
                                FUN7(VAR39,VAR42[VAR15].FUN36(VAR10)));
            } else {
                VAR10 &= 3;
                FUN35(VAR47[0], VAR48,
                                FUN7(VAR39,VAR40[VAR15].VAR41.FUN37(VAR10)));
            }
            break;
        case 0xc5: 
        case 0x1c5:
            if (VAR13 != 3)
                goto VAR24;
            VAR11 = (VAR2->VAR44 == 2) ? VAR45 : VAR49;
            VAR10 = FUN4(VAR2->VAR36++);
            if (VAR6) {
                VAR10 &= 7;
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN38(VAR47[0], VAR48,
                                 FUN7(VAR39,VAR42[VAR14].FUN36(VAR10)));
            } else {
                VAR10 &= 3;
                VAR14 = (VAR12 & 7);
                FUN38(VAR47[0], VAR48,
                                FUN7(VAR39,VAR40[VAR14].VAR41.FUN37(VAR10)));
            }
            VAR15 = ((VAR12 >> 3) & 7) | VAR5;
            FUN34(VAR11, VAR15);
            break;
        case 0x1d6: 
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                FUN6(VAR2->VAR38, FUN7(VAR39,VAR42[VAR15].FUN25(0)));
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN26(FUN7(VAR39,VAR42[VAR14].FUN25(0)),
                            FUN7(VAR39,VAR42[VAR15].FUN25(0)));
                FUN29(FUN7(VAR39,VAR42[VAR14].FUN25(1)));
            }
            break;
        case 0x2d6: 
            FUN3(VAR35);
            VAR14 = (VAR12 & 7);
            FUN26(FUN7(VAR39,VAR42[VAR15].FUN25(0)),
                        FUN7(VAR39,VAR40[VAR14].VAR41));
            FUN29(FUN7(VAR39,VAR42[VAR15].FUN25(1)));
            break;
        case 0x3d6: 
            FUN3(VAR35);
            VAR14 = (VAR12 & 7) | FUN18(VAR2);
            FUN26(FUN7(VAR39,VAR40[VAR15 & 7].VAR41),
                        FUN7(VAR39,VAR42[VAR14].FUN25(0)));
            break;
        case 0xd7: 
        case 0x1d7:
            if (VAR13 != 3)
                goto VAR24;
            if (VAR6) {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                FUN11(VAR50, VAR48, FUN7(VAR39,VAR42[VAR14]));
                FUN32(VAR71, VAR53, VAR50);
            } else {
                VAR14 = (VAR12 & 7);
                FUN11(VAR50, VAR48, FUN7(VAR39,VAR40[VAR14].VAR41));
                FUN32(VAR72, VAR53, VAR50);
            }
            FUN33(VAR47[0], VAR53);
            VAR15 = ((VAR12 >> 3) & 7) | VAR5;
            FUN34(VAR49, VAR15);
            break;
        case 0x038:
        case 0x138:
            VAR3 = VAR12;
            VAR12 = FUN4(VAR2->VAR36++);
            VAR14 = VAR12 & 7;
            VAR15 = ((VAR12 >> 3) & 7) | VAR5;
            VAR13 = (VAR12 >> 6) & 3;
            if (VAR2->VAR19 & VAR22)
                goto VAR73;
            VAR18 = VAR74[VAR3].VAR75[VAR6];
            if (!VAR18)
                goto VAR24;
            if (!(VAR2->VAR76 & VAR74[VAR3].VAR77))
                goto VAR24;
            if (VAR6) {
                VAR7 = FUN7(VAR39,VAR42[VAR15]);
                if (VAR13 == 3) {
                    VAR8 = FUN7(VAR39,VAR42[VAR14 | FUN18(VAR2)]);
                } else {
                    VAR8 = FUN7(VAR39,VAR56);
                    FUN5(VAR2, VAR12, &VAR16, &VAR17);
                    switch (VAR3) {
                    case 0x20: case 0x30: 
                    case 0x23: case 0x33: 
                    case 0x25: case 0x35: 
                        FUN14(VAR2->VAR38, VAR8 +
                                        FUN7(VAR78, FUN25(0)));
                        break;
                    case 0x21: case 0x31: 
                    case 0x24: case 0x34: 
                        FUN39(VAR53, VAR79,
                                          (VAR2->VAR38 >> 2) - 1);
                        FUN40(VAR53, VAR48, VAR8 +
                                        FUN7(VAR78, FUN22(0)));
                        break;
                    case 0x22: case 0x32: 
                        FUN41(VAR80, VAR79,
                                          (VAR2->VAR38 >> 2) - 1);
                        FUN35(VAR80, VAR48, VAR8 +
                                        FUN7(VAR78, FUN36(0)));
                        break;
                    case 0x2a:            
                        FUN17(VAR2->VAR38, VAR7);
                        return;
                    default:
                        FUN17(VAR2->VAR38, VAR8);
                    }
                }
            } else {
                VAR7 = FUN7(VAR39,VAR40[VAR15].VAR41);
                if (VAR13 == 3) {
                    VAR8 = FUN7(VAR39,VAR40[VAR14].VAR41);
                } else {
                    VAR8 = FUN7(VAR39,VAR57);
                    FUN5(VAR2, VAR12, &VAR16, &VAR17);
                    FUN14(VAR2->VAR38, VAR8);
                }
            }
            if (VAR18 == VAR37)
                goto VAR24;
            FUN11(VAR50, VAR48, VAR7);
            FUN11(VAR59, VAR48, VAR8);
            FUN12(VAR18, VAR50, VAR59);
            if (VAR3 == 0x17)
                VAR2->VAR81 = VAR82;
            break;
        case 0x338: 
        VAR73:
            VAR3 = VAR12;
            VAR12 = FUN4(VAR2->VAR36++);
            VAR15 = ((VAR12 >> 3) & 7) | VAR5;
            if (VAR3 != 0xf0 && VAR3 != 0xf1)
                goto VAR24;
            if (!(VAR2->VAR76 & VAR83))
                goto VAR24;
            if (VAR3 == 0xf0)
                VAR11 = VAR84;
            else if (VAR3 == 0xf1 && VAR2->VAR44 != 2)
                if (VAR2->VAR19 & VAR20)
                    VAR11 = VAR70;
                else
                    VAR11 = VAR49;
            else
                VAR11 = VAR45;
            FUN42(VAR49, 0, VAR15);
            FUN13(VAR53, VAR47[0]);
            FUN9(VAR2, VAR12, VAR11, VAR46, 0);
            FUN43(VAR85, VAR47[0], VAR53,
                            VAR47[0], FUN44(8 << VAR11));
            VAR11 = (VAR2->VAR44 == 2) ? VAR45 : VAR49;
            FUN34(VAR11, VAR15);
            break;
        case 0x03a:
        case 0x13a:
            VAR3 = VAR12;
            VAR12 = FUN4(VAR2->VAR36++);
            VAR14 = VAR12 & 7;
            VAR15 = ((VAR12 >> 3) & 7) | VAR5;
            VAR13 = (VAR12 >> 6) & 3;
            VAR18 = VAR86[VAR3].VAR75[VAR6];
            if (!VAR18)
                goto VAR24;
            if (!(VAR2->VAR76 & VAR86[VAR3].VAR77))
                goto VAR24;
            if (VAR18 == VAR37) {
                VAR11 = (VAR2->VAR44 == 2) ? VAR45 : VAR49;
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                if (VAR13 != 3)
                    FUN5(VAR2, VAR12, &VAR16, &VAR17);
                VAR15 = ((VAR12 >> 3) & 7) | VAR5;
                VAR10 = FUN4(VAR2->VAR36++);
                switch (VAR3) {
                case 0x14: 
                    FUN45(VAR47[0], VAR48, FUN7(VAR39,
                                            VAR42[VAR15].FUN46(VAR10 & 15)));
                    if (VAR13 == 3)
                        FUN34(VAR11, VAR14);
                    else
                        FUN47(VAR47[0], VAR79,
                                        (VAR2->VAR38 >> 2) - 1);
                    break;
                case 0x15: 
                    FUN38(VAR47[0], VAR48, FUN7(VAR39,
                                            VAR42[VAR15].FUN36(VAR10 & 7)));
                    if (VAR13 == 3)
                        FUN34(VAR11, VAR14);
                    else
                        FUN48(VAR47[0], VAR79,
                                        (VAR2->VAR38 >> 2) - 1);
                    break;
                case 0x16:
                    if (VAR11 == VAR49) { 
                        FUN49(VAR53, VAR48,
                                        FUN7(VAR39,
                                                VAR42[VAR15].FUN22(VAR10 & 3)));
                        if (VAR13 == 3)
                            FUN50(VAR11, VAR14, VAR53);
                        else
                            FUN51(VAR53, VAR79,
                                            (VAR2->VAR38 >> 2) - 1);
                    } else { 
                        FUN15(VAR55, VAR48,
                                        FUN7(VAR39,
                                                VAR42[VAR15].FUN25(VAR10 & 1)));
                        if (VAR13 == 3)
                            FUN50(VAR11, VAR14, VAR55);
                        else
                            FUN52(VAR55, VAR79,
                                            (VAR2->VAR38 >> 2) - 1);
                    }
                    break;
                case 0x17: 
                    FUN27(VAR47[0], VAR48, FUN7(VAR39,
                                            VAR42[VAR15].FUN22(VAR10 & 3)));
                    if (VAR13 == 3)
                        FUN34(VAR11, VAR14);
                    else
                        FUN51(VAR47[0], VAR79,
                                        (VAR2->VAR38 >> 2) - 1);
                    break;
                case 0x20: 
                    if (VAR13 == 3)
                        FUN42(VAR49, 0, VAR14);
                    else
                        FUN53(VAR47[0], VAR79,
                                        (VAR2->VAR38 >> 2) - 1);
                    FUN54(VAR47[0], VAR48, FUN7(VAR39,
                                            VAR42[VAR15].FUN46(VAR10 & 15)));
                    break;
                case 0x21: 
                    if (VAR13 == 3)
                        FUN49(VAR53, VAR48,
                                        FUN7(VAR39,VAR42[VAR14]
                                                .FUN22((VAR10 >> 6) & 3)));
                    else
                        FUN39(VAR53, VAR79,
                                        (VAR2->VAR38 >> 2) - 1);
                    FUN40(VAR53, VAR48,
                                    FUN7(VAR39,VAR42[VAR15]
                                            .FUN22((VAR10 >> 4) & 3)));
                    if ((VAR10 >> 0) & 1)
                        FUN40(FUN44(0 ),
                                        VAR48, FUN7(VAR39,
                                                VAR42[VAR15].FUN22(0)));
                    if ((VAR10 >> 1) & 1)
                        FUN40(FUN44(0 ),
                                        VAR48, FUN7(VAR39,
                                                VAR42[VAR15].FUN22(1)));
                    if ((VAR10 >> 2) & 1)
                        FUN40(FUN44(0 ),
                                        VAR48, FUN7(VAR39,
                                                VAR42[VAR15].FUN22(2)));
                    if ((VAR10 >> 3) & 1)
                        FUN40(FUN44(0 ),
                                        VAR48, FUN7(VAR39,
                                                VAR42[VAR15].FUN22(3)));
                    break;
                case 0x22:
                    if (VAR11 == VAR49) { 
                        if (VAR13 == 3)
                            FUN55(VAR11, VAR53, VAR14);
                        else
                            FUN39(VAR53, VAR79,
                                            (VAR2->VAR38 >> 2) - 1);
                        FUN40(VAR53, VAR48,
                                        FUN7(VAR39,
                                                VAR42[VAR15].FUN22(VAR10 & 3)));
                    } else { 
                        if (VAR13 == 3)
                            FUN55(VAR11, VAR55, VAR14);
                        else
                            FUN56(VAR55, VAR79,
                                            (VAR2->VAR38 >> 2) - 1);
                        FUN16(VAR55, VAR48,
                                        FUN7(VAR39,
                                                VAR42[VAR15].FUN25(VAR10 & 1)));
                    }
                    break;
                }
                return;
            }
            if (VAR6) {
                VAR7 = FUN7(VAR39,VAR42[VAR15]);
                if (VAR13 == 3) {
                    VAR8 = FUN7(VAR39,VAR42[VAR14 | FUN18(VAR2)]);
                } else {
                    VAR8 = FUN7(VAR39,VAR56);
                    FUN5(VAR2, VAR12, &VAR16, &VAR17);
                    FUN17(VAR2->VAR38, VAR8);
                }
            } else {
                VAR7 = FUN7(VAR39,VAR40[VAR15].VAR41);
                if (VAR13 == 3) {
                    VAR8 = FUN7(VAR39,VAR40[VAR14].VAR41);
                } else {
                    VAR8 = FUN7(VAR39,VAR57);
                    FUN5(VAR2, VAR12, &VAR16, &VAR17);
                    FUN14(VAR2->VAR38, VAR8);
                }
            }
            VAR10 = FUN4(VAR2->VAR36++);
            if ((VAR3 & 0xfc) == 0x60) { 
                VAR2->VAR81 = VAR82;
                if (VAR2->VAR44 == 2)
                    
                    VAR10 |= 1 << 8;
            }
            FUN11(VAR50, VAR48, VAR7);
            FUN11(VAR59, VAR48, VAR8);
            FUN57(VAR18, VAR50, VAR59, FUN44(VAR10));
            break;
        default:
            goto VAR24;
        }
    } else {
        
        switch(VAR3) {
        case 0x70: 
        case 0xc6: 
        case 0xc2: 
            VAR2->VAR69 = 1;
            break;
        default:
            break;
        }
        if (VAR9) {
            VAR7 = FUN7(VAR39,VAR42[VAR15]);
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                VAR8 = FUN7(VAR39,VAR56);
                if (VAR6 >= 2 && ((VAR3 >= 0x50 && VAR3 <= 0x5f && VAR3 != 0x5b) ||
                                VAR3 == 0xc2)) {
                    
                    if (VAR6 == 2) {
                        
                        FUN20(VAR49 + VAR2->VAR38);
                        FUN21(VAR47[0], VAR48, FUN7(VAR39,VAR56.FUN22(0)));
                    } else {
                        
                        FUN14(VAR2->VAR38, FUN7(VAR39,VAR56.FUN58(0)));
                    }
                } else {
                    FUN17(VAR2->VAR38, VAR8);
                }
            } else {
                VAR14 = (VAR12 & 7) | FUN18(VAR2);
                VAR8 = FUN7(VAR39,VAR42[VAR14]);
            }
        } else {
            VAR7 = FUN7(VAR39,VAR40[VAR15].VAR41);
            if (VAR13 != 3) {
                FUN5(VAR2, VAR12, &VAR16, &VAR17);
                VAR8 = FUN7(VAR39,VAR57);
                FUN14(VAR2->VAR38, VAR8);
            } else {
                VAR14 = (VAR12 & 7);
                VAR8 = FUN7(VAR39,VAR40[VAR14].VAR41);
            }
        }
        switch(VAR3) {
        case 0x0f: 
            if (!(VAR2->VAR32 & VAR33))
                goto VAR24;
            VAR10 = FUN4(VAR2->VAR36++);
            VAR18 = VAR87[VAR10];
            if (!VAR18)
                goto VAR24;
            FUN11(VAR50, VAR48, VAR7);
            FUN11(VAR59, VAR48, VAR8);
            FUN12(VAR18, VAR50, VAR59);
            break;
        case 0x70: 
        case 0xc6: 
            VAR10 = FUN4(VAR2->VAR36++);
            FUN11(VAR50, VAR48, VAR7);
            FUN11(VAR59, VAR48, VAR8);
            FUN57(VAR18, VAR50, VAR59, FUN44(VAR10));
            break;
        case 0xc2:
            
            VAR10 = FUN4(VAR2->VAR36++);
            if (VAR10 >= 8)
                goto VAR24;
            VAR18 = VAR88[VAR10][VAR6];
            FUN11(VAR50, VAR48, VAR7);
            FUN11(VAR59, VAR48, VAR8);
            FUN12(VAR18, VAR50, VAR59);
            break;
        case 0xf7:
            
            if (VAR13 != 3)
                goto VAR24;
#ifdef VAR43
            if (VAR2->VAR89 == 2) {
                FUN59(VAR90);
            } else
#endif
            {
                FUN60(VAR90);
                if (VAR2->VAR89 == 0)
                    FUN61();
            }
            FUN62(VAR2);
            FUN11(VAR50, VAR48, VAR7);
            FUN11(VAR59, VAR48, VAR8);
            FUN57(VAR18, VAR50, VAR59, VAR79);
            break;
        default:
            FUN11(VAR50, VAR48, VAR7);
            FUN11(VAR59, VAR48, VAR8);
            FUN12(VAR18, VAR50, VAR59);
            break;
        }
        if (VAR3 == 0x2e || VAR3 == 0x2f) {
            VAR2->VAR81 = VAR82;
        }
    }
}